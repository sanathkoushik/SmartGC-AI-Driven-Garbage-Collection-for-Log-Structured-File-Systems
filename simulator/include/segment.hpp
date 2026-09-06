#ifndef SMARTGC_SEGMENT_HPP
#define SMARTGC_SEGMENT_HPP

#include "types.hpp"
#include "block.hpp"
#include <vector>
#include <cstddef>

namespace smartgc {

class Segment {
public:
    Segment(size_t id, size_t capacity);

    size_t id() const { return segment_id_; }
    size_t capacity() const { return capacity_; }
    size_t valid_count() const { return valid_count_; }
    size_t invalid_count() const { return invalid_count_; }
    size_t free_count() const { return free_count_; }
    size_t next_write_index() const { return next_write_index_; }
    bool is_open() const { return is_open_; }
    bool is_full() const { return next_write_index_ >= capacity_; }
    bool has_free_block() const { return next_write_index_ < capacity_ && free_count_ > 0; }

    void set_open(bool open) { is_open_ = open; }

    const Block& get_block(size_t block_index) const;
    const std::vector<Block>& blocks() const { return blocks_; }

    // Sequential write into the segment
    size_t allocate_block(LbaType lba);

    // Invalidate a previously written block
    void invalidate_block(size_t block_index);

    // Erase/reset segment to FREE state
    void reset();

    double valid_ratio() const;
    double invalid_ratio() const;

private:
    size_t segment_id_;
    size_t capacity_;
    std::vector<Block> blocks_;
    size_t next_write_index_ = 0;
    size_t valid_count_ = 0;
    size_t invalid_count_ = 0;
    size_t free_count_ = 0;
    bool is_open_ = false;
};

} // namespace smartgc

#endif // SMARTGC_SEGMENT_HPP
