#ifndef SMARTGC_LFS_SIMULATOR_HPP
#define SMARTGC_LFS_SIMULATOR_HPP

#include "types.hpp"
#include "config.hpp"
#include "segment.hpp"
#include "mapping.hpp"
#include "metrics.hpp"
#include <array>
#include <vector>
#include <deque>
#include <string>

namespace smartgc {

// Deterministic log-structured storage engine with greedy segment cleaning.
//
// Invariants maintained at every observable point (see tests/test_lfs.cpp):
//   1. mapping().size() == number of VALID blocks across all segments.
//   2. Every mapped LBA resolves to a VALID block holding exactly that LBA.
//   3. physical_bytes_written == logical_bytes_written + gc_bytes_copied.
//   4. Cleaning never migrates an INVALID block and never selects an open
//      segment or a fully valid segment as a victim.
//   5. User writes never drive the free pool below config.gc_reserved_segments,
//      so a cleaning pass can always place the victim's surviving blocks.
class LfsSimulator {
public:
    explicit LfsSimulator(const SimulatorConfig& config = SimulatorConfig{});

    void init();
    void reset();

    // Writes one block-sized update for `lba`.
    //
    // `size_bytes` must equal config.block_size_bytes: normalized traces are
    // expanded to one record per 4KB block during preprocessing, so a request
    // spanning several blocks must never reach this call as a single write.
    // `temperature` is a placement hint only; it has no effect on validity.
    // Throws DeviceFullError when the pool cannot hold another live block.
    bool write(LbaType lba, size_t size_bytes, Temperature temperature = Temperature::UNKNOWN);
    bool write(LbaType lba) { return write(lba, config_.block_size_bytes, Temperature::UNKNOWN); }

    bool read(LbaType lba) const;

    // Runs one greedy cleaning pass. Returns false when no victim can free
    // space (no closed segment exists, or every closed segment is fully valid).
    bool run_greedy_gc();
    // Cleans only if the free pool has fallen to the configured watermark.
    bool trigger_gc_if_needed();

    const StorageMetrics& metrics() const { return metrics_; }
    const SimulatorConfig& config() const { return config_; }
    const LbaMapping& mapping() const { return lba_mapping_; }
    const std::vector<Segment>& segments() const { return segments_; }
    size_t free_segment_count() const { return free_segment_ids_.size(); }

    // Id of the segment currently receiving writes for `stream`, or SIZE_MAX
    // when that stream has no open segment yet.
    size_t open_segment_id(WriteStream stream) const;
    size_t live_blocks() const { return lba_mapping_.size(); }

    size_t count_total_valid_blocks() const;
    size_t count_total_invalid_blocks() const;
    size_t count_total_free_blocks() const;
    double utilization() const;
    void print_status_summary() const;

private:
    struct OpenSegment {
        size_t id = 0;
        bool active = false;
    };

    WriteStream select_user_stream(Temperature temperature) const;
    // True when `stream` has an open segment with room for another block.
    bool stream_writable(WriteStream stream) const;

    // Makes `stream` ready to accept a block, cleaning first if the free pool is
    // at the watermark. Only used on the user write path.
    void ensure_user_stream_writable(WriteStream stream);
    // Makes `stream` ready during a cleaning pass. Never recurses into cleaning
    // and is allowed to draw on the GC reserve.
    void ensure_gc_stream_writable(WriteStream stream);
    // Detaches the currently open segment of `stream`, if any.
    void close_stream(WriteStream stream);
    // Pops a segment from the free pool and opens it for `stream`.
    void open_segment_for(WriteStream stream);

    bool select_greedy_victim(size_t& victim_id) const;
    void migrate_valid_block(LbaType lba);
    void place_block(WriteStream stream, LbaType lba);

    SimulatorConfig config_;
    std::vector<Segment> segments_;
    std::deque<size_t> free_segment_ids_;
    std::array<OpenSegment, NUM_WRITE_STREAMS> open_segments_{};

    LbaMapping lba_mapping_;
    StorageMetrics metrics_;
};

} // namespace smartgc

#endif // SMARTGC_LFS_SIMULATOR_HPP
