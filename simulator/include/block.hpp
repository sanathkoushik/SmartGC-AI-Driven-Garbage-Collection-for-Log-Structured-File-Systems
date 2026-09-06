#ifndef SMARTGC_BLOCK_HPP
#define SMARTGC_BLOCK_HPP

#include "types.hpp"

namespace smartgc {

struct Block {
    LbaType lba = INVALID_LBA;
    BlockState state = BlockState::FREE;

    bool is_free() const { return state == BlockState::FREE; }
    bool is_valid() const { return state == BlockState::VALID; }
    bool is_invalid() const { return state == BlockState::INVALID; }

    void allocate(LbaType new_lba) {
        lba = new_lba;
        state = BlockState::VALID;
    }

    void invalidate() {
        state = BlockState::INVALID;
    }

    void reset() {
        lba = INVALID_LBA;
        state = BlockState::FREE;
    }
};

} // namespace smartgc

#endif // SMARTGC_BLOCK_HPP
