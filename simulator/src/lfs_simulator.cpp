#include "lfs_simulator.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <chrono>

namespace smartgc {

LfsSimulator::LfsSimulator(const SimulatorConfig& config)
    : config_(config) {
    init();
}

void LfsSimulator::init() {
    segments_.clear();
    free_segment_ids_.clear();
    lba_mapping_.clear();
    metrics_.reset();

    if (config_.total_segments == 0 || config_.blocks_per_segment == 0) {
        throw std::invalid_argument("Invalid simulator configuration: segments and blocks must be > 0");
    }

    stream_count_ = config_.effective_stream_count();
    if (stream_count_ < 1) stream_count_ = 1;
    if (stream_count_ > MAX_STREAM_CLASSES) stream_count_ = MAX_STREAM_CLASSES;
    stream_head_ids_.assign(stream_count_, 0);
    stream_head_active_.assign(stream_count_, 0);

    write_event_index_ = 0;
    last_pred_stream_.clear();
    last_write_event_.clear();
    write_count_.clear();
    gc_decision_latency_sum_us_ = 0.0;
    gc_decision_count_ = 0;
    user_writes_at_last_gc_ = 0;

    trigger_controller_.configure(config_.learned_trigger,
                                  config_.gc_free_segments_threshold,
                                  config_.total_segments,
                                  config_.gc_policy_table_path);

    segments_.reserve(config_.total_segments);
    for (size_t i = 0; i < config_.total_segments; ++i) {
        segments_.emplace_back(i, config_.blocks_per_segment);
        free_segment_ids_.push_back(i);
    }

    // Allocate the primary (stream 0) open segment eagerly, matching Phase 1.
    ensure_stream_head_available(0);
}

void LfsSimulator::reset() {
    init();
}

void LfsSimulator::load_predictions(const std::string& path) {
    predictions_.load(path);
}

// -----------------------------------------------------------------------------
// GC trigger
// -----------------------------------------------------------------------------
GcTriggerState LfsSimulator::build_trigger_state() const {
    GcTriggerState s;
    s.free_segment_ratio = config_.total_segments == 0
        ? 1.0
        : static_cast<double>(free_segment_ids_.size()) / static_cast<double>(config_.total_segments);
    const uint64_t window = config_.trigger_state_window_events == 0 ? 1 : config_.trigger_state_window_events;
    const uint64_t recent = metrics_.total_write_requests - user_writes_at_last_gc_;
    s.recent_write_rate = std::min(1.0, static_cast<double>(recent) / static_cast<double>(window));
    s.recent_waf = metrics_.waf();
    return s;
}

// Reserve the free pool must be kept above. With N concurrent log heads a
// single GC pass can, worst case, pop up to N fresh segments for migration
// destinations before reclaiming its victim, so multi-stream runs need one
// free segment per head plus a margin. The repeated-cleanup loop in
// ensure_stream_head_available() then restores a healthy pool.
size_t LfsSimulator::gc_reserve_target() const {
    if (stream_count_ <= 1) return config_.gc_free_segments_threshold;
    return std::max(stream_count_ + 1, config_.gc_free_segments_threshold);
}

bool LfsSimulator::gc_should_trigger() {
    const GcTriggerState st = build_trigger_state();
    const auto t0 = std::chrono::steady_clock::now();
    bool decision = trigger_controller_.should_trigger(free_segment_ids_.size(), st);
    // Hard safety floor: for multi-stream runs keep enough headroom that a GC
    // pass can always place its migrated blocks. Single-stream keeps the exact
    // Phase 1 rule (controller only).
    if (stream_count_ > 1 && free_segment_ids_.size() <= gc_reserve_target()) {
        decision = true;
    }
    const auto t1 = std::chrono::steady_clock::now();
    gc_decision_latency_sum_us_ +=
        std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(t1 - t0).count();
    gc_decision_count_++;
    return decision;
}

// -----------------------------------------------------------------------------
// Open-segment management (per stream log head)
// -----------------------------------------------------------------------------
void LfsSimulator::ensure_stream_head_available(size_t stream) {
    if (stream >= stream_count_) stream = 0;

    // 1. Current head still has room.
    if (stream_head_active_[stream] &&
        segments_[stream_head_ids_[stream]].has_free_block()) {
        return;
    }

    // 2. Head is full -> close it.
    if (stream_head_active_[stream]) {
        segments_[stream_head_ids_[stream]].set_open(false);
        stream_head_active_[stream] = 0;
    }

    // 3. Trigger GC if the controller says so. For multi-stream runs keep
    //    cleaning until the free pool is back above the reserve target (a GC
    //    pass can transiently consume segments for migration destinations, so a
    //    single pass is not guaranteed to net-free space).
    if (gc_should_trigger()) {
        run_greedy_gc();
        if (stream_count_ > 1) {
            const size_t target = gc_reserve_target() + 1;
            size_t stall_guard = 0;
            while (free_segment_ids_.size() < target && stall_guard < 2 * config_.total_segments) {
                const size_t before = free_segment_ids_.size();
                if (!run_greedy_gc()) break;
                if (free_segment_ids_.size() <= before) ++stall_guard; else stall_guard = 0;
            }
        }
    }

    // 4. GC may have re-opened this head during migration.
    if (stream_head_active_[stream] &&
        segments_[stream_head_ids_[stream]].has_free_block()) {
        return;
    }

    // 5. Keep running GC while the free pool is empty.
    while (free_segment_ids_.empty()) {
        if (!run_greedy_gc()) {
            throw std::runtime_error("LFS Simulator Out of Space: Cannot allocate segment and GC cannot free any space.");
        }
        if (stream_head_active_[stream] &&
            segments_[stream_head_ids_[stream]].has_free_block()) {
            return;
        }
    }

    // 6. Pop the next free segment for this head.
    const size_t seg_id = free_segment_ids_.front();
    free_segment_ids_.pop_front();
    stream_head_ids_[stream] = seg_id;
    segments_[seg_id].set_open(true);
    stream_head_active_[stream] = 1;
}

// -----------------------------------------------------------------------------
// Placement / migration stream decisions
// -----------------------------------------------------------------------------

// Map a semantic lifetime class onto a physical log-head index for the current
// stream count: 1 head -> everything to 0; 2 heads -> SHORT hot / else cold;
// 3 heads -> SHORT/MEDIUM/LONG map straight through.
size_t LfsSimulator::phys_stream(StreamClass c) const {
    if (stream_count_ <= 1) return 0;
    if (stream_count_ == 2) return (c == StreamClass::SHORT) ? 0 : 1;
    size_t idx = static_cast<size_t>(c);
    if (idx >= stream_count_) idx = stream_count_ - 1;
    return idx;
}

StreamClass LfsSimulator::rule_based_class(LbaType lba) const {
    // Zero-training heuristic: an LBA is "hot" (SHORT) once it has been rewritten
    // and its most recent observed inter-write gap is short.
    auto wc = write_count_.find(lba);
    if (wc == write_count_.end() || wc->second < 2) {
        return StreamClass::LONG; // unseen -> assume cold
    }
    auto lw = last_write_event_.find(lba);
    if (lw == last_write_event_.end()) {
        return StreamClass::LONG;
    }
    const size_t observed_interval = write_event_index_ - lw->second;
    const size_t threshold = config_.blocks_per_segment; // ~ one segment of intervening writes
    return observed_interval <= threshold ? StreamClass::SHORT : StreamClass::LONG;
}

size_t LfsSimulator::decide_placement_stream(LbaType lba) {
    switch (config_.placement_policy) {
        case PlacementPolicy::MIXED:
            return 0;

        case PlacementPolicy::SUP_LIKE:
            // SUP-GC: every fresh write defaults to the hot (SHORT) stream.
            return phys_stream(StreamClass::SHORT);

        case PlacementPolicy::RULE_BASED:
            return phys_stream(rule_based_class(lba));

        case PlacementPolicy::STAT_ML:
        case PlacementPolicy::LSTM_SMARTGC:
        case PlacementPolicy::LSTM_ATTN_SMARTGC:
        case PlacementPolicy::HYBRID_ROBUST_SMARTGC: {
            if (!predictions_.has_row(write_event_index_)) {
                return phys_stream(rule_based_class(lba)); // heuristic fallback
            }
            const Prediction& p = predictions_.row(write_event_index_);
            metrics_.predicted_writes++;
            StreamClass sc;
            if (config_.confidence_fallback_threshold > 0.0 &&
                p.confidence < config_.confidence_fallback_threshold) {
                metrics_.confidence_fallbacks++;
                sc = rule_based_class(lba);
            } else {
                sc = p.stream(stream_count_);
            }
            last_pred_stream_[lba] = static_cast<int>(sc); // remember for migration routing
            return phys_stream(sc);
        }
        default:
            return 0;
    }
}

size_t LfsSimulator::decide_migration_stream(LbaType lba) const {
    if (stream_count_ == 1) return 0;

    switch (config_.placement_policy) {
        case PlacementPolicy::SUP_LIKE:
            // SUP-GC: GC-recovered valid blocks default to the cold (LONG) stream.
            return phys_stream(StreamClass::LONG);

        case PlacementPolicy::RULE_BASED:
            return phys_stream(rule_based_class(lba));

        case PlacementPolicy::STAT_ML:
        case PlacementPolicy::LSTM_SMARTGC:
        case PlacementPolicy::LSTM_ATTN_SMARTGC:
        case PlacementPolicy::HYBRID_ROBUST_SMARTGC: {
            auto it = last_pred_stream_.find(lba);
            if (it != last_pred_stream_.end()) {
                return phys_stream(static_cast<StreamClass>(
                    it->second < 0 ? 0 : (it->second > 2 ? 2 : it->second)));
            }
            // Survived a GC pass with no prediction on record -> treat as colder.
            return phys_stream(stream_count_ >= 3 ? StreamClass::MEDIUM : StreamClass::LONG);
        }
        default:
            return 0;
    }
}

// -----------------------------------------------------------------------------
// Core write path
// -----------------------------------------------------------------------------
bool LfsSimulator::write(LbaType lba, size_t /*size_bytes*/) {
    if (lba < 0) {
        return false;
    }

    const size_t target_stream = decide_placement_stream(lba);

    ensure_stream_head_available(target_stream);

    // 1. Invalidate previous copy if the LBA was already mapped.
    PhysicalAddress old_addr;
    if (lba_mapping_.get(lba, old_addr)) {
        segments_[old_addr.segment_id].invalidate_block(old_addr.block_index);
    }

    // 2. Allocate sequentially in the target stream's open segment.
    const size_t head = stream_head_ids_[target_stream];
    const size_t new_block_idx = segments_[head].allocate_block(lba);

    // 3. Update L2P mapping.
    lba_mapping_.set(lba, PhysicalAddress{head, new_block_idx});

    // 4. Metrics (user logical write + physical write).
    metrics_.logical_bytes_written += config_.block_size_bytes;
    metrics_.physical_bytes_written += config_.block_size_bytes;
    metrics_.total_write_requests++;

    // 5. Per-LBA bookkeeping for heuristics / migration routing.
    write_count_[lba]++;
    last_write_event_[lba] = write_event_index_;
    write_event_index_++;

    return true;
}

bool LfsSimulator::read(LbaType lba) const {
    return lba_mapping_.contains(lba);
}

bool LfsSimulator::trigger_gc_if_needed() {
    if (gc_should_trigger()) {
        return run_greedy_gc();
    }
    return false;
}

// -----------------------------------------------------------------------------
// Greedy GC
// -----------------------------------------------------------------------------
bool LfsSimulator::select_greedy_victim(size_t& victim_id) const {
    bool found = false;
    size_t min_valid_blocks = std::numeric_limits<size_t>::max();
    size_t max_invalid_blocks = 0;

    for (size_t i = 0; i < segments_.size(); ++i) {
        const auto& seg = segments_[i];

        // Skip any currently open log head.
        if (seg.is_open()) {
            continue;
        }

        // Skip completely unwritten / already free segments.
        if (seg.next_write_index() == 0 && seg.free_count() == seg.capacity()) {
            continue;
        }

        if (!found ||
            seg.valid_count() < min_valid_blocks ||
            (seg.valid_count() == min_valid_blocks && seg.invalid_count() > max_invalid_blocks)) {
            victim_id = i;
            found = true;
            min_valid_blocks = seg.valid_count();
            max_invalid_blocks = seg.invalid_count();
        }
    }

    return found;
}

void LfsSimulator::migrate_valid_block(LbaType lba, size_t dest_stream) {
    if (dest_stream >= stream_count_) dest_stream = 0;

    if (!stream_head_active_[dest_stream] ||
        !segments_[stream_head_ids_[dest_stream]].has_free_block()) {
        if (stream_head_active_[dest_stream]) {
            segments_[stream_head_ids_[dest_stream]].set_open(false);
            stream_head_active_[dest_stream] = 0;
        }
        if (free_segment_ids_.empty()) {
            throw std::runtime_error("GC migration failed: No free segment available to receive migrated valid block");
        }
        const size_t seg_id = free_segment_ids_.front();
        free_segment_ids_.pop_front();
        stream_head_ids_[dest_stream] = seg_id;
        segments_[seg_id].set_open(true);
        stream_head_active_[dest_stream] = 1;
    }

    const size_t head = stream_head_ids_[dest_stream];
    const size_t new_block_idx = segments_[head].allocate_block(lba);
    lba_mapping_.set(lba, PhysicalAddress{head, new_block_idx});

    // GC migrations contribute to physical writes only, never logical.
    metrics_.physical_bytes_written += config_.block_size_bytes;
    metrics_.gc_bytes_copied += config_.block_size_bytes;
    metrics_.valid_blocks_migrated++;
}

bool LfsSimulator::run_greedy_gc() {
    size_t victim_id = 0;
    if (!select_greedy_victim(victim_id)) {
        return false;
    }

    Segment& victim = segments_[victim_id];

    std::vector<LbaType> valid_lbas_to_migrate;
    for (size_t b = 0; b < victim.capacity(); ++b) {
        const auto& block = victim.get_block(b);
        if (block.is_valid()) {
            valid_lbas_to_migrate.push_back(block.lba);
        }
    }

    for (LbaType lba : valid_lbas_to_migrate) {
        migrate_valid_block(lba, decide_migration_stream(lba));
    }

    victim.reset();
    free_segment_ids_.push_back(victim_id);
    metrics_.gc_count++;
    metrics_.per_gc_migrated.push_back(static_cast<uint64_t>(valid_lbas_to_migrate.size()));
    user_writes_at_last_gc_ = metrics_.total_write_requests;

    return true;
}

// -----------------------------------------------------------------------------
// Inspection
// -----------------------------------------------------------------------------
size_t LfsSimulator::count_total_valid_blocks() const {
    size_t total = 0;
    for (const auto& seg : segments_) total += seg.valid_count();
    return total;
}

size_t LfsSimulator::count_total_invalid_blocks() const {
    size_t total = 0;
    for (const auto& seg : segments_) total += seg.invalid_count();
    return total;
}

size_t LfsSimulator::count_total_free_blocks() const {
    size_t total = 0;
    for (const auto& seg : segments_) total += seg.free_count();
    return total;
}

void LfsSimulator::print_status_summary() const {
    std::cout << "\n================ Storage Pool Status ================\n";
    std::cout << " Total Segments:        " << config_.total_segments << "\n";
    std::cout << " Blocks Per Segment:    " << config_.blocks_per_segment << "\n";
    std::cout << " Free Segments in Pool: " << free_segment_ids_.size() << "\n";
    std::cout << " Write Streams:         " << stream_count_ << " [";
    for (size_t s = 0; s < stream_count_; ++s) {
        std::cout << (s ? ", " : "") << stream_class_to_string(static_cast<StreamClass>(s))
                  << "=seg" << stream_head_ids_[s];
    }
    std::cout << "]\n";
    std::cout << " Learned GC Trigger:    " << (trigger_controller_.learned() ? "ON" : "OFF") << "\n";
    std::cout << " Mapped Active LBAs:    " << lba_mapping_.size() << "\n";
    std::cout << " Total Valid Blocks:    " << count_total_valid_blocks() << "\n";
    std::cout << " Total Invalid Blocks:  " << count_total_invalid_blocks() << "\n";
    std::cout << " Total Free Blocks:     " << count_total_free_blocks() << "\n";
    std::cout << "-----------------------------------------------------\n";
    for (size_t i = 0; i < segments_.size(); ++i) {
        const auto& seg = segments_[i];
        std::cout << " Segment " << std::setw(2) << i << ": ["
                  << "V:" << std::setw(3) << seg.valid_count() << " | "
                  << "I:" << std::setw(3) << seg.invalid_count() << " | "
                  << "F:" << std::setw(3) << seg.free_count() << "] "
                  << (seg.is_open() ? " (OPEN)" : "") << "\n";
    }
    std::cout << "=====================================================\n\n";
}

} // namespace smartgc
