#include "lfs_simulator.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <limits>

namespace smartgc {
namespace {

constexpr size_t stream_index(WriteStream s) {
    return static_cast<size_t>(s);
}

} // namespace

LfsSimulator::LfsSimulator(const SimulatorConfig& config)
    : config_(config) {
    config_.validate();
    init();
}

void LfsSimulator::init() {
    segments_.clear();
    free_segment_ids_.clear();
    lba_mapping_.clear();
    metrics_.reset();
    open_segments_.fill(OpenSegment{});

    segments_.reserve(config_.total_segments);
    for (size_t i = 0; i < config_.total_segments; ++i) {
        segments_.emplace_back(i, config_.blocks_per_segment);
        free_segment_ids_.push_back(i);
    }
}

void LfsSimulator::reset() {
    init();
}

size_t LfsSimulator::open_segment_id(WriteStream stream) const {
    const OpenSegment& os = open_segments_[stream_index(stream)];
    return os.active ? os.id : std::numeric_limits<size_t>::max();
}

WriteStream LfsSimulator::select_user_stream(Temperature temperature) const {
    if (config_.placement_policy == PlacementPolicy::MIXED) {
        return WriteStream::MAIN;
    }
    // Segregated placement: predicted-cold data is kept out of the hot log so it
    // does not survive repeated cleaning passes alongside short-lived blocks.
    // A write with no prediction falls back to the main log, which is the
    // conservative choice (it never strands unknown data in the cold segment).
    return temperature == Temperature::COLD ? WriteStream::COLD : WriteStream::MAIN;
}

void LfsSimulator::close_stream(WriteStream stream) {
    OpenSegment& os = open_segments_[stream_index(stream)];
    if (os.active) {
        segments_[os.id].set_open(false);
        os.active = false;
    }
}

bool LfsSimulator::stream_writable(WriteStream stream) const {
    const OpenSegment& os = open_segments_[stream_index(stream)];
    return os.active && segments_[os.id].has_free_block();
}

void LfsSimulator::open_segment_for(WriteStream stream) {
    if (free_segment_ids_.empty()) {
        throw std::logic_error("open_segment_for called with an empty free pool");
    }
    OpenSegment& os = open_segments_[stream_index(stream)];
    if (os.active) {
        // Replacing a live append point would strand its segment: it would stay
        // flagged open forever and could never be selected as a victim.
        throw std::logic_error("open_segment_for called on a stream that is already open");
    }
    os.id = free_segment_ids_.front();
    free_segment_ids_.pop_front();
    os.active = true;
    segments_[os.id].set_open(true);
}

void LfsSimulator::ensure_user_stream_writable(WriteStream stream) {
    if (stream_writable(stream)) {
        return;
    }
    close_stream(stream);

    // A user write may never consume the GC reserve: keep cleaning until the
    // pool holds more than both the watermark and the reserve.
    const size_t floor_segments = std::max(config_.gc_free_segments_threshold,
                                           config_.gc_reserved_segments);
    // A victim is never fully valid, so a pass migrates at most capacity - 1
    // blocks and frees a whole segment: total free blocks strictly increase by
    // at least one per pass. The loop therefore cannot run longer than the block
    // capacity of the device. This bound is a livelock guard, not a policy - a
    // healthy configuration exits after a handful of passes.
    const size_t max_passes = config_.total_capacity_blocks() + 1;
    size_t passes = 0;

    while (free_segment_ids_.size() <= floor_segments) {
        const bool cleaned = run_greedy_gc();
        if (!cleaned || ++passes > max_passes) {
            std::ostringstream oss;
            oss << "Device full: cleaning cannot free a segment for a user write ("
                << (cleaned ? "no progress after " + std::to_string(passes) + " passes"
                            : "no reclaimable victim")
                << "; live blocks " << live_blocks() << " of " << config_.max_live_blocks()
                << " usable, raw capacity " << config_.total_capacity_blocks()
                << " blocks, free segments " << free_segment_ids_.size() << ").";
            throw DeviceFullError(oss.str());
        }
    }

    // When GC shares this append point (gc_separate_stream = false) a cleaning
    // pass above may already have opened a segment for it. Adopting that segment
    // is required for correctness: opening a second one here would leave the
    // first flagged open forever, permanently excluding it from cleaning.
    if (stream_writable(stream)) {
        return;
    }
    open_segment_for(stream);
}

void LfsSimulator::ensure_gc_stream_writable(WriteStream stream) {
    if (stream_writable(stream)) {
        return;
    }
    close_stream(stream);
    if (free_segment_ids_.empty()) {
        // Unreachable while gc_reserved_segments >= 1 and user writes respect the
        // reserve: a victim holds at most blocks_per_segment - 1 valid blocks, so
        // one reserved segment always absorbs a pass.
        throw std::logic_error("GC reserve exhausted during migration; the reserve invariant is broken");
    }
    open_segment_for(stream);
}

void LfsSimulator::place_block(WriteStream stream, LbaType lba) {
    const OpenSegment& os = open_segments_[stream_index(stream)];
    const size_t block_index = segments_[os.id].allocate_block(lba);
    lba_mapping_.set(lba, PhysicalAddress{os.id, block_index});
}

bool LfsSimulator::write(LbaType lba, size_t size_bytes, Temperature temperature) {
    if (lba < 0) {
        return false;
    }
    if (size_bytes != config_.block_size_bytes) {
        std::ostringstream oss;
        oss << "LfsSimulator::write expects exactly one block (" << config_.block_size_bytes
            << " bytes) but received " << size_bytes
            << " bytes. Multi-block requests must be expanded into per-block records "
               "during preprocessing.";
        throw std::invalid_argument(oss.str());
    }

    const bool is_new_lba = !lba_mapping_.contains(lba);
    if (is_new_lba && live_blocks() >= config_.max_live_blocks()) {
        std::ostringstream oss;
        oss << "Device full: cannot admit new LBA " << lba << "; live blocks ("
            << live_blocks() << ") already at the usable limit of "
            << config_.max_live_blocks() << " blocks ("
            << std::fixed << std::setprecision(2) << (config_.max_utilization() * 100.0)
            << "% of raw capacity). Reduce the working set, add segments, or lower "
               "simulator.gc_reserved_segments.";
        throw DeviceFullError(oss.str());
    }

    const WriteStream stream = select_user_stream(temperature);
    ensure_user_stream_writable(stream);

    // Invalidate the previous copy *after* cleaning, so a block that this write
    // is about to obsolete may still be migrated. That is the behaviour of a real
    // device, which cannot see the future.
    PhysicalAddress old_addr;
    if (lba_mapping_.get(lba, old_addr)) {
        segments_[old_addr.segment_id].invalidate_block(old_addr.block_index);
    }

    place_block(stream, lba);

    metrics_.logical_bytes_written += config_.block_size_bytes;
    metrics_.physical_bytes_written += config_.block_size_bytes;
    metrics_.total_write_requests++;
    if (stream == WriteStream::COLD) {
        metrics_.cold_stream_writes++;
    } else {
        metrics_.main_stream_writes++;
    }
    if (temperature == Temperature::UNKNOWN) {
        metrics_.unpredicted_writes++;
    }

    return true;
}

bool LfsSimulator::read(LbaType lba) const {
    return lba_mapping_.contains(lba);
}

bool LfsSimulator::trigger_gc_if_needed() {
    if (free_segment_ids_.size() <= config_.gc_free_segments_threshold) {
        return run_greedy_gc();
    }
    return false;
}

bool LfsSimulator::select_greedy_victim(size_t& victim_id) const {
    bool found = false;
    size_t best_valid = 0;
    size_t best_invalid = 0;

    for (size_t i = 0; i < segments_.size(); ++i) {
        const Segment& seg = segments_[i];

        // Never clean a segment that is currently an append point.
        if (seg.is_open()) {
            continue;
        }
        // Never clean a segment that holds no data (it is already in the pool).
        if (seg.next_write_index() == 0) {
            continue;
        }
        // A fully valid segment yields no free space, so cleaning it is pure
        // write amplification with no progress.
        if (seg.valid_count() == seg.capacity()) {
            continue;
        }

        const size_t valid = seg.valid_count();
        const size_t invalid = seg.invalid_count();
        const bool better = !found ||
                            valid < best_valid ||
                            (valid == best_valid && invalid > best_invalid);
        if (better) {
            victim_id = i;
            best_valid = valid;
            best_invalid = invalid;
            found = true;
        }
    }

    return found;
}

void LfsSimulator::migrate_valid_block(LbaType lba) {
    const WriteStream stream = config_.gc_separate_stream ? WriteStream::GC : WriteStream::MAIN;
    ensure_gc_stream_writable(stream);
    place_block(stream, lba);

    // Migrations are physical writes only. Logical bytes describe the workload
    // and must not grow when the device cleans itself.
    metrics_.physical_bytes_written += config_.block_size_bytes;
    metrics_.gc_bytes_copied += config_.block_size_bytes;
    metrics_.valid_blocks_migrated++;
}

bool LfsSimulator::run_greedy_gc() {
    size_t victim_id = 0;
    if (!select_greedy_victim(victim_id)) {
        return false;
    }

    // Snapshot the surviving LBAs before any migration mutates the victim's
    // neighbours, so the set cleaned is exactly the set observed.
    std::vector<LbaType> surviving;
    surviving.reserve(segments_[victim_id].valid_count());
    for (size_t b = 0; b < segments_[victim_id].capacity(); ++b) {
        const Block& block = segments_[victim_id].get_block(b);
        if (block.is_valid()) {
            surviving.push_back(block.lba);
        }
    }

    for (const LbaType lba : surviving) {
        migrate_valid_block(lba);
    }

    segments_[victim_id].reset();
    free_segment_ids_.push_back(victim_id);
    metrics_.gc_count++;
    return true;
}

size_t LfsSimulator::count_total_valid_blocks() const {
    size_t total = 0;
    for (const Segment& seg : segments_) {
        total += seg.valid_count();
    }
    return total;
}

size_t LfsSimulator::count_total_invalid_blocks() const {
    size_t total = 0;
    for (const Segment& seg : segments_) {
        total += seg.invalid_count();
    }
    return total;
}

size_t LfsSimulator::count_total_free_blocks() const {
    size_t total = 0;
    for (const Segment& seg : segments_) {
        total += seg.free_count();
    }
    return total;
}

double LfsSimulator::utilization() const {
    const size_t cap = config_.total_capacity_blocks();
    return cap == 0 ? 0.0 : static_cast<double>(live_blocks()) / static_cast<double>(cap);
}

void LfsSimulator::print_status_summary() const {
    std::cout << "\n================ Storage Pool Status ================\n";
    std::cout << " Total Segments:        " << config_.total_segments << "\n";
    std::cout << " Blocks Per Segment:    " << config_.blocks_per_segment << "\n";
    std::cout << " Free Segments in Pool: " << free_segment_ids_.size()
              << " (reserve " << config_.gc_reserved_segments << ")\n";
    for (size_t s = 0; s < NUM_WRITE_STREAMS; ++s) {
        const WriteStream stream = static_cast<WriteStream>(s);
        const size_t id = open_segment_id(stream);
        std::cout << " Open Segment [" << std::setw(4) << write_stream_to_string(stream) << "]: ";
        if (id == std::numeric_limits<size_t>::max()) {
            std::cout << "none\n";
        } else {
            std::cout << id << "\n";
        }
    }
    std::cout << " Mapped Active LBAs:    " << lba_mapping_.size() << "\n";
    std::cout << " Total Valid Blocks:    " << count_total_valid_blocks() << "\n";
    std::cout << " Total Invalid Blocks:  " << count_total_invalid_blocks() << "\n";
    std::cout << " Total Free Blocks:     " << count_total_free_blocks() << "\n";
    std::cout << " Live Utilization:      " << std::fixed << std::setprecision(2)
              << (utilization() * 100.0) << "% of raw capacity\n";
    std::cout << "=====================================================\n\n";
}

} // namespace smartgc
