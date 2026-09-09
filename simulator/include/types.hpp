#ifndef SMARTGC_TYPES_HPP
#define SMARTGC_TYPES_HPP

#include <cstdint>
#include <cstddef>
#include <stdexcept>
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

inline bool parse_placement_policy(const std::string& text, PlacementPolicy& out) {
    if (text == "MIXED") { out = PlacementPolicy::MIXED; return true; }
    if (text == "RULE_BASED") { out = PlacementPolicy::RULE_BASED; return true; }
    if (text == "LSTM_SMARTGC") { out = PlacementPolicy::LSTM_SMARTGC; return true; }
    return false;
}

// Predicted temperature of an incoming write. This is a *placement* hint only and
// is deliberately independent of block validity (see docs/architecture.md).
enum class Temperature : uint8_t {
    COLD = 0,
    HOT = 1,
    UNKNOWN = 2   // no prediction available; policy decides the fallback stream
};

inline const char* temperature_to_string(Temperature t) {
    switch (t) {
        case Temperature::COLD: return "COLD";
        case Temperature::HOT: return "HOT";
        case Temperature::UNKNOWN: return "UNKNOWN";
        default: return "UNKNOWN";
    }
}

// Independent append points into the log.
//   MAIN - carries every user write under MIXED, and the predicted-HOT user
//          writes under the segregated placement policies.
//   COLD - carries predicted-COLD user writes; unused under MIXED.
//   GC   - carries blocks migrated out of victim segments during cleaning, so
//          that surviving (long-lived) data is not interleaved into a user log.
enum class WriteStream : uint8_t {
    MAIN = 0,
    COLD = 1,
    GC = 2
};

constexpr size_t NUM_WRITE_STREAMS = 3;

inline const char* write_stream_to_string(WriteStream s) {
    switch (s) {
        case WriteStream::MAIN: return "MAIN";
        case WriteStream::COLD: return "COLD";
        case WriteStream::GC: return "GC";
        default: return "UNKNOWN";
    }
}

// Raised when the storage pool cannot accept another live block, or when garbage
// collection can no longer reclaim space because every closed segment is fully
// valid. Distinct from std::runtime_error so experiments can report it precisely
// instead of mistaking it for an internal fault.
class DeviceFullError : public std::runtime_error {
public:
    explicit DeviceFullError(const std::string& what) : std::runtime_error(what) {}
};

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
