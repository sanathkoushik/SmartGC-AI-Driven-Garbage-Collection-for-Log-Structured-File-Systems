#ifndef SMARTGC_PREDICTIONS_HPP
#define SMARTGC_PREDICTIONS_HPP

#include "types.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace smartgc {

// One row of the predictions.csv contract (v2).
//   timestamp,lba,predicted_rewrite_interval,predicted_class,predicted_stream_class,confidence
// v1 files (first four columns only) are still accepted: predicted_stream_class
// is then derived from predicted_class and confidence defaults to 1.0.
struct Prediction {
    LbaType lba = INVALID_LBA;
    double predicted_rewrite_interval = 0.0;
    int predicted_class = 0;         // 1 = HOT, 0 = COLD
    int predicted_stream_class = -1; // 0..2 bucket; -1 => derive from predicted_class
    double confidence = 1.0;         // [0, 1]; 1.0 when a producer omits it

    StreamClass stream(size_t stream_count) const {
        if (predicted_stream_class >= 0) {
            return stream_from_bucket(predicted_stream_class, stream_count);
        }
        return stream_from_binary_class(predicted_class);
    }
};

// Batch offline predictions consumed by the simulator in lockstep with the
// trace: row i corresponds to the i-th write event replayed.
class PredictionTable {
public:
    PredictionTable() = default;

    void load(const std::string& path) {
        rows_.clear();
        std::ifstream in(path);
        if (!in.is_open()) {
            throw std::runtime_error("PredictionTable: cannot open predictions file: " + path);
        }

        std::string line;
        if (!std::getline(in, line)) {
            throw std::runtime_error("PredictionTable: empty predictions file: " + path);
        }
        const std::vector<std::string> header = split_csv(line);
        const int idx_lba = find_col(header, "lba");
        const int idx_interval = find_col(header, "predicted_rewrite_interval");
        const int idx_class = find_col(header, "predicted_class");
        const int idx_stream = find_col(header, "predicted_stream_class");
        const int idx_conf = find_col(header, "confidence");
        if (idx_lba < 0 || idx_class < 0) {
            throw std::runtime_error("PredictionTable: missing required columns (lba, predicted_class) in " + path);
        }

        while (std::getline(in, line)) {
            if (line.empty()) continue;
            const std::vector<std::string> f = split_csv(line);
            if (static_cast<int>(f.size()) <= idx_lba) continue;

            Prediction p;
            p.lba = static_cast<LbaType>(std::stoll(f[idx_lba]));
            if (idx_interval >= 0 && idx_interval < static_cast<int>(f.size()) && !f[idx_interval].empty()) {
                p.predicted_rewrite_interval = std::stod(f[idx_interval]);
            }
            p.predicted_class = parse_class(f[idx_class]);
            if (idx_stream >= 0 && idx_stream < static_cast<int>(f.size()) && !f[idx_stream].empty()) {
                p.predicted_stream_class = std::stoi(f[idx_stream]);
            }
            if (idx_conf >= 0 && idx_conf < static_cast<int>(f.size()) && !f[idx_conf].empty()) {
                p.confidence = std::stod(f[idx_conf]);
            }
            rows_.push_back(p);
        }
    }

    bool empty() const { return rows_.empty(); }
    size_t size() const { return rows_.size(); }
    bool has_row(size_t idx) const { return idx < rows_.size(); }
    const Prediction& row(size_t idx) const { return rows_.at(idx); }

private:
    static std::vector<std::string> split_csv(const std::string& line) {
        std::vector<std::string> out;
        std::string cur;
        std::istringstream ss(line);
        while (std::getline(ss, cur, ',')) {
            // trim whitespace / CR
            size_t b = cur.find_first_not_of(" \t\r\n");
            size_t e = cur.find_last_not_of(" \t\r\n");
            out.push_back(b == std::string::npos ? std::string() : cur.substr(b, e - b + 1));
        }
        return out;
    }

    static int find_col(const std::vector<std::string>& header, const std::string& name) {
        for (size_t i = 0; i < header.size(); ++i) {
            std::string h = header[i];
            std::transform(h.begin(), h.end(), h.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (h == name) return static_cast<int>(i);
        }
        return -1;
    }

    static int parse_class(const std::string& v) {
        std::string s = v;
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (s == "hot" || s == "1" || s == "true") return 1;
        if (s == "cold" || s == "0" || s == "false") return 0;
        // numeric fallthrough (e.g. "1.0")
        try { return std::stod(s) >= 0.5 ? 1 : 0; } catch (...) { return 0; }
    }

    std::vector<Prediction> rows_;
};

} // namespace smartgc

#endif // SMARTGC_PREDICTIONS_HPP
