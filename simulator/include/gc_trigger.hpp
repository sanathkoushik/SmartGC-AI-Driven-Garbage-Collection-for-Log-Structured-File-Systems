#ifndef SMARTGC_GC_TRIGGER_HPP
#define SMARTGC_GC_TRIGGER_HPP

#include "types.hpp"
#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>

namespace smartgc {

// Contextual state exposed to the (optional) learned GC trigger controller.
// Mirrors the state vector used by ml/training/gc_controller.py:
//   [free_segment_ratio, recent_write_rate, recent_waf]
struct GcTriggerState {
    double free_segment_ratio = 1.0; // free_segments / total_segments
    double recent_write_rate = 0.0;  // user writes in the last decision window / window size
    double recent_waf = 1.0;         // WAF observed so far
};

// Decides whether GC should run now. Two modes:
//   * fixed   - reproduces the Phase 1 low-watermark rule exactly.
//   * learned - discretizes the state and looks up an argmax action in a
//               Q-table (loaded from CSV). When no table is supplied a
//               deterministic adaptive rule is used so runs stay
//               reproducible and unit-testable without Python in the loop.
class GcTriggerController {
public:
    GcTriggerController() = default;

    void configure(bool learned, size_t fixed_watermark, size_t total_segments,
                   const std::string& q_table_path) {
        learned_ = learned;
        fixed_watermark_ = fixed_watermark;
        total_segments_ = total_segments == 0 ? 1 : total_segments;
        have_q_table_ = false;
        q_table_.clear();
        if (learned_ && !q_table_path.empty()) {
            load_q_table(q_table_path);
        }
    }

    bool learned() const { return learned_; }

    // Returns true if GC should be triggered given the current state.
    bool should_trigger(size_t free_segments, const GcTriggerState& s) const {
        if (!learned_) {
            return free_segments <= fixed_watermark_;
        }
        // Never let the pool run dry regardless of the policy.
        if (free_segments <= 1) return true;

        if (have_q_table_) {
            const int state_id = discretize(s);
            const auto it = q_table_.find(state_id);
            if (it != q_table_.end()) {
                // action 1 == trigger_now, action 0 == wait
                return it->second[1] >= it->second[0];
            }
        }
        // Deterministic adaptive fallback: raise the effective watermark when a
        // write burst is in progress or write amplification is climbing, so GC
        // runs earlier and keeps more headroom; relax it when the system is calm.
        double effective = static_cast<double>(fixed_watermark_);
        if (s.recent_write_rate > 0.75) effective += 2.0;
        else if (s.recent_write_rate > 0.5) effective += 1.0;
        if (s.recent_waf > 1.15) effective += 1.0;
        if (s.recent_write_rate < 0.15 && s.recent_waf < 1.05) effective -= 1.0;
        if (effective < 1.0) effective = 1.0;
        return static_cast<double>(free_segments) <= effective;
    }

private:
    // 3x3x3 discretization of the continuous state, matching the Python trainer.
    int discretize(const GcTriggerState& s) const {
        const int a = bucket3(s.free_segment_ratio, 0.10, 0.30);
        const int b = bucket3(s.recent_write_rate, 0.33, 0.66);
        const int c = bucket3(s.recent_waf - 1.0, 0.05, 0.15);
        return a * 9 + b * 3 + c;
    }

    static int bucket3(double v, double lo, double hi) {
        if (v < lo) return 0;
        if (v < hi) return 1;
        return 2;
    }

    void load_q_table(const std::string& path) {
        std::ifstream in(path);
        if (!in.is_open()) return;
        std::string line;
        std::getline(in, line); // header: state_id,q_wait,q_trigger
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            std::string tok;
            std::vector<std::string> f;
            while (std::getline(ss, tok, ',')) f.push_back(tok);
            if (f.size() < 3) continue;
            const int sid = std::stoi(f[0]);
            std::array<double, 2> qa{{std::stod(f[1]), std::stod(f[2])}};
            q_table_[sid] = qa;
        }
        have_q_table_ = !q_table_.empty();
    }

    bool learned_ = false;
    size_t fixed_watermark_ = 2;
    size_t total_segments_ = 1;
    bool have_q_table_ = false;
    std::unordered_map<int, std::array<double, 2>> q_table_;
};

} // namespace smartgc

#endif // SMARTGC_GC_TRIGGER_HPP
