#ifndef SMARTGC_TYPES_HPP
#define SMARTGC_TYPES_HPP

#include <cstdint>
#include <cstddef>
#include <string>

namespace smartgc {

using LbaType = int64_t;
constexpr LbaType INVALID_LBA = -1;

enum class BlockState : uint8_t {
    FREE = 0,
    VALID = 1,
    INVALID = 2
};

inline const char* block_state_to_string(BlockState state) {
    switch (state) {
        case BlockState::FREE: return "FREE";
        case BlockState::VALID: return "VALID";
        case BlockState::INVALID: return "INVALID";
        default: return "UNKNOWN";
    }
}

enum class GcPolicy : uint8_t {
    GREEDY = 0
};

inline const char* gc_policy_to_string(GcPolicy policy) {
    switch (policy) {
        case GcPolicy::GREEDY: return "GREEDY";
        default: return "UNKNOWN";
    }
}

enum class PlacementPolicy : uint8_t {
    MIXED = 0,
    RULE_BASED = 1,
    LSTM_SMARTGC = 2
};

inline const char* placement_policy_to_string(PlacementPolicy policy) {
    switch (policy) {
        case PlacementPolicy::MIXED: return "MIXED";
        case PlacementPolicy::RULE_BASED: return "RULE_BASED";
        case PlacementPolicy::LSTM_SMARTGC: return "LSTM_SMARTGC";
        default: return "UNKNOWN";
    }
}

struct PhysicalAddress {
    size_t segment_id;
    size_t block_index;

    bool operator==(const PhysicalAddress& other) const {
        return segment_id == other.segment_id && block_index == other.block_index;
    }

    bool operator!=(const PhysicalAddress& other) const {
        return !(*this == other);
    }
};

} // namespace smartgc

#endif // SMARTGC_TYPES_HPP
