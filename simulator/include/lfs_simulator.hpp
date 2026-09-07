#ifndef SMARTGC_LFS_SIMULATOR_HPP
#define SMARTGC_LFS_SIMULATOR_HPP

#include "types.hpp"
#include "config.hpp"
#include "segment.hpp"
#include "mapping.hpp"
#include "metrics.hpp"
#include "predictions.hpp"
#include "gc_trigger.hpp"
#include <vector>
#include <deque>
#include <string>
#include <unordered_map>

namespace smartgc {

class LfsSimulator {
public:
    explicit LfsSimulator(const SimulatorConfig& config = SimulatorConfig{});

    // Initialize/reset storage pool and log structures
    void init();
    void reset();

    // Optional batch predictions consumed in lockstep with write events.
    void load_predictions(const std::string& path);
    bool has_predictions() const { return !predictions_.empty(); }

    // Core I/O operations
    bool write(LbaType lba, size_t size_bytes = 4096);
    bool read(LbaType lba) const;

    // Garbage collection
    bool trigger_gc_if_needed();
    bool run_greedy_gc();

    // Inspection & status
    const StorageMetrics& metrics() const { return metrics_; }
    const SimulatorConfig& config() const { return config_; }
    const LbaMapping& mapping() const { return lba_mapping_; }
    const std::vector<Segment>& segments() const { return segments_; }
    size_t free_segment_count() const { return free_segment_ids_.size(); }
    size_t stream_count() const { return stream_count_; }
    size_t stream_head_id(size_t stream) const { return stream_head_ids_.at(stream); }
    // Backward-compatible accessor: the primary (stream 0) log head.
    size_t current_open_segment_id() const { return stream_head_ids_.empty() ? 0 : stream_head_ids_[0]; }
    double avg_gc_decision_latency_us() const {
        return gc_decision_count_ == 0 ? 0.0
               : gc_decision_latency_sum_us_ / static_cast<double>(gc_decision_count_);
    }

    // Helpers
    size_t count_total_valid_blocks() const;
    size_t count_total_invalid_blocks() const;
    size_t count_total_free_blocks() const;
    void print_status_summary() const;

private:
    void ensure_stream_head_available(size_t stream);
    bool select_greedy_victim(size_t& victim_id) const;
    void migrate_valid_block(LbaType lba, size_t dest_stream);

    // Placement / migration stream decisions.
    size_t decide_placement_stream(LbaType lba);
    size_t decide_migration_stream(LbaType lba) const;
    StreamClass rule_based_class(LbaType lba) const;   // zero-training hot/cold heuristic
    size_t phys_stream(StreamClass c) const;           // semantic class -> physical head index

    // Learned / fixed GC trigger.
    GcTriggerState build_trigger_state() const;
    bool gc_should_trigger();
    size_t gc_reserve_target() const;

    SimulatorConfig config_;
    std::vector<Segment> segments_;
    std::deque<size_t> free_segment_ids_;

    size_t stream_count_ = 1;
    std::vector<size_t> stream_head_ids_;   // one open log head per stream class
    std::vector<char> stream_head_active_;

    LbaMapping lba_mapping_;
    StorageMetrics metrics_;

    // Prediction ingestion + per-LBA memory for migration-time routing.
    PredictionTable predictions_;
    size_t write_event_index_ = 0;
    std::unordered_map<LbaType, int> last_pred_stream_;   // last predicted bucket per LBA
    std::unordered_map<LbaType, size_t> last_write_event_; // event index of previous write
    std::unordered_map<LbaType, uint32_t> write_count_;    // writes seen per LBA

    // GC trigger controller state.
    GcTriggerController trigger_controller_;
    double gc_decision_latency_sum_us_ = 0.0;
    uint64_t gc_decision_count_ = 0;
    uint64_t user_writes_at_last_gc_ = 0;
};

} // namespace smartgc

#endif // SMARTGC_LFS_SIMULATOR_HPP
