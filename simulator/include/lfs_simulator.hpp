#ifndef SMARTGC_LFS_SIMULATOR_HPP
#define SMARTGC_LFS_SIMULATOR_HPP

#include "types.hpp"
#include "config.hpp"
#include "segment.hpp"
#include "mapping.hpp"
#include "metrics.hpp"
#include <vector>
#include <deque>
#include <string>

namespace smartgc {

class LfsSimulator {
public:
    explicit LfsSimulator(const SimulatorConfig& config = SimulatorConfig{});

    // Initialize/reset storage pool and log structures
    void init();
    void reset();

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
    size_t current_open_segment_id() const { return current_open_segment_id_; }

    // Helpers
    size_t count_total_valid_blocks() const;
    size_t count_total_invalid_blocks() const;
    size_t count_total_free_blocks() const;
    void print_status_summary() const;

private:
    // Internal helper to get/advance the open segment
    void ensure_open_segment_available();
    bool select_greedy_victim(size_t& victim_id) const;
    void migrate_valid_block(LbaType lba);

    SimulatorConfig config_;
    std::vector<Segment> segments_;
    std::deque<size_t> free_segment_ids_;
    size_t current_open_segment_id_ = 0;
    bool has_active_open_segment_ = false;

    LbaMapping lba_mapping_;
    StorageMetrics metrics_;
};

} // namespace smartgc

#endif // SMARTGC_LFS_SIMULATOR_HPP
