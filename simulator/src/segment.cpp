#include "segment.hpp"
#include <stdexcept>

namespace smartgc {

Segment::Segment(size_t id, size_t capacity)
    : segment_id_(id),
      capacity_(capacity),
      blocks_(capacity),
      next_write_index_(0),
      valid_count_(0),
      invalid_count_(0),
      free_count_(capacity),
      is_open_(false) {}

const Block& Segment::get_block(size_t block_index) const {
    if (block_index >= capacity_) {
        throw std::out_of_range("Block index out of range in segment " + std::to_string(segment_id_));
    }
    return blocks_[block_index];
}

size_t Segment::allocate_block(LbaType lba) {
    if (!has_free_block()) {
        throw std::runtime_error("Cannot allocate block: Segment " + std::to_string(segment_id_) + " is full");
    }

    size_t allocated_idx = next_write_index_++;
    blocks_[allocated_idx].allocate(lba);
    valid_count_++;
    free_count_--;

    return allocated_idx;
}

void Segment::invalidate_block(size_t block_index) {
    if (block_index >= capacity_) {
        throw std::out_of_range("Invalidate: Block index out of range in segment " + std::to_string(segment_id_));
    }

    if (blocks_[block_index].is_valid()) {
        blocks_[block_index].invalidate();
        valid_count_--;
        invalid_count_++;
    }
}

void Segment::reset() {
    for (auto& block : blocks_) {
        block.reset();
    }
    next_write_index_ = 0;
    valid_count_ = 0;
    invalid_count_ = 0;
    free_count_ = capacity_;
    is_open_ = false;
}

double Segment::valid_ratio() const {
    return capacity_ == 0 ? 0.0 : static_cast<double>(valid_count_) / static_cast<double>(capacity_);
}

double Segment::invalid_ratio() const {
    return capacity_ == 0 ? 0.0 : static_cast<double>(invalid_count_) / static_cast<double>(capacity_);
}

} // namespace smartgc
