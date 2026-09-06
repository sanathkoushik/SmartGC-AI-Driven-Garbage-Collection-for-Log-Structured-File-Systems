#include "lfs_simulator.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <limits>

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
    has_active_open_segment_ = false;

    if (config_.total_segments == 0 || config_.blocks_per_segment == 0) {
        throw std::invalid_argument("Invalid simulator configuration: segments and blocks must be > 0");
    }

    segments_.reserve(config_.total_segments);
    for (size_t i = 0; i < config_.total_segments; ++i) {
        segments_.emplace_back(i, config_.blocks_per_segment);
        free_segment_ids_.push_back(i);
    }

    // Allocate initial open segment
    ensure_open_segment_available();
}

void LfsSimulator::reset() {
    init();
}

void LfsSimulator::ensure_open_segment_available() {
    // 1. If currently open segment has free space, we are ready to write
    if (has_active_open_segment_ && segments_[current_open_segment_id_].has_free_block()) {
        return;
    }

    // 2. If current segment is full, close it
    if (has_active_open_segment_) {
        segments_[current_open_segment_id_].set_open(false);
        has_active_open_segment_ = false;
    }

    // 3. Trigger GC if free segments pool is at or below threshold
    if (free_segment_ids_.size() <= config_.gc_free_segments_threshold) {
        run_greedy_gc();
    }

    // 4. If GC opened a segment during migration and it still has free blocks, reuse it
    if (has_active_open_segment_ && segments_[current_open_segment_id_].has_free_block()) {
        return;
    }

    // 5. If free pool is empty, keep running GC until a segment is freed
    while (free_segment_ids_.empty()) {
        if (!run_greedy_gc()) {
            throw std::runtime_error("LFS Simulator Out of Space: Cannot allocate segment and GC cannot free any space.");
        }
        if (has_active_open_segment_ && segments_[current_open_segment_id_].has_free_block()) {
            return;
        }
    }

    // 6. Pop next free segment from pool
    current_open_segment_id_ = free_segment_ids_.front();
    free_segment_ids_.pop_front();
    segments_[current_open_segment_id_].set_open(true);
    has_active_open_segment_ = true;
}

bool LfsSimulator::write(LbaType lba, size_t /*size_bytes*/) {
    if (lba < 0) {
        return false;
    }

    // Ensure we have an open segment with available free block
    ensure_open_segment_available();

    // 1. Invalidate previous copy if LBA was already mapped
    PhysicalAddress old_addr;
    if (lba_mapping_.get(lba, old_addr)) {
        segments_[old_addr.segment_id].invalidate_block(old_addr.block_index);
    }

    // 2. Allocate block sequentially in current open segment
    size_t new_block_idx = segments_[current_open_segment_id_].allocate_block(lba);

    // 3. Update L2P mapping table
    lba_mapping_.set(lba, PhysicalAddress{current_open_segment_id_, new_block_idx});

    // 4. Update metrics (user logical write + physical write)
    metrics_.logical_bytes_written += config_.block_size_bytes;
    metrics_.physical_bytes_written += config_.block_size_bytes;
    metrics_.total_write_requests++;

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
    size_t min_valid_blocks = std::numeric_limits<size_t>::max();
    size_t max_invalid_blocks = 0;

    for (size_t i = 0; i < segments_.size(); ++i) {
        const auto& seg = segments_[i];

        // Skip currently open segment
        if (seg.is_open()) {
            continue;
        }

        // Skip completely unwritten / already free segments
        if (seg.next_write_index() == 0 && seg.free_count() == seg.capacity()) {
            continue;
        }

        // Greedy policy: pick segment with fewest valid blocks
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

void LfsSimulator::migrate_valid_block(LbaType lba) {
    // Ensure open segment has room for migrated block
    if (!has_active_open_segment_ || !segments_[current_open_segment_id_].has_free_block()) {
        if (has_active_open_segment_) {
            segments_[current_open_segment_id_].set_open(false);
            has_active_open_segment_ = false;
        }

        if (free_segment_ids_.empty()) {
            throw std::runtime_error("GC migration failed: No free segment available to receive migrated valid block");
        }

        current_open_segment_id_ = free_segment_ids_.front();
        free_segment_ids_.pop_front();
        segments_[current_open_segment_id_].set_open(true);
        has_active_open_segment_ = true;
    }

    size_t new_block_idx = segments_[current_open_segment_id_].allocate_block(lba);
    lba_mapping_.set(lba, PhysicalAddress{current_open_segment_id_, new_block_idx});

    // Accounting: physical write and GC migration count increment, logical write DOES NOT increment
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

    // Collect valid blocks to migrate
    std::vector<LbaType> valid_lbas_to_migrate;
    for (size_t b = 0; b < victim.capacity(); ++b) {
        const auto& block = victim.get_block(b);
        if (block.is_valid()) {
            valid_lbas_to_migrate.push_back(block.lba);
        }
    }

    // Migrate valid blocks out of the victim segment
    for (LbaType lba : valid_lbas_to_migrate) {
        migrate_valid_block(lba);
    }

    // Reclaim/erase the victim segment
    victim.reset();
    free_segment_ids_.push_back(victim_id);
    metrics_.gc_count++;

    return true;
}

size_t LfsSimulator::count_total_valid_blocks() const {
    size_t total = 0;
    for (const auto& seg : segments_) {
        total += seg.valid_count();
    }
    return total;
}

size_t LfsSimulator::count_total_invalid_blocks() const {
    size_t total = 0;
    for (const auto& seg : segments_) {
        total += seg.invalid_count();
    }
    return total;
}

size_t LfsSimulator::count_total_free_blocks() const {
    size_t total = 0;
    for (const auto& seg : segments_) {
        total += seg.free_count();
    }
    return total;
}

void LfsSimulator::print_status_summary() const {
    std::cout << "\n================ Storage Pool Status ================\n";
    std::cout << " Total Segments:        " << config_.total_segments << "\n";
    std::cout << " Blocks Per Segment:    " << config_.blocks_per_segment << "\n";
    std::cout << " Free Segments in Pool: " << free_segment_ids_.size() << "\n";
    std::cout << " Active Open Segment:   " << current_open_segment_id_ << "\n";
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
