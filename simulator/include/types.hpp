#ifndef SMARTGC_TYPES_HPP
#define SMARTGC_TYPES_HPP

#include <cstdint>
#include <cstddef>
#include <string>
#include <stdexcept>

namespace smartgc {

using LbaType = int64_t;
constexpr LbaType INVALID_LBA = -1;

// Interface contract version shared by the CSV producers/consumers.
// Bumped from the implicit v1 (Phase 1) when Phase 4b/5 added
// predicted_stream_class / confidence to predictions and stream/trigger
// counters to the metrics contract.
constexpr int CONTRACT_VERSION = 2;

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

// Placement policy for incoming writes.
//   MIXED                 - baseline: every write enters one shared log head.
//   RULE_BASED            - zero-training heuristic: threshold on recent observed interval.
//   LSTM_SMARTGC          - PyTorch LSTM regressor -> HOT/COLD (2-way) placement.
//   SUP_LIKE              - SUP-GC inspired: fresh writes default HOT, GC copies default COLD.
//   STAT_ML               - lightweight sklearn classifier (logistic / GBDT) on engineered features.
//   LSTM_ATTN_SMARTGC     - LSTM backbone + additive attention -> multi-level stream buckets.
//   HYBRID_ROBUST_SMARTGC - Phase 8: same LSTM+attention model, but the Python
//                           export layer blends its prediction with RULE_BASED
//                           continuously (confidence-weighted, capped by a
//                           robustness parameter) instead of a binary gate;
//                           the simulator consumes the resulting stream class
//                           exactly like LSTM_ATTN_SMARTGC (confidence-based
//                           re-gating is disabled for this policy since the
//                           blend already accounts for confidence upstream).
enum class PlacementPolicy : uint8_t {
    MIXED = 0,
    RULE_BASED = 1,
    LSTM_SMARTGC = 2,
    SUP_LIKE = 3,
    STAT_ML = 4,
    LSTM_ATTN_SMARTGC = 5,
    HYBRID_ROBUST_SMARTGC = 6
};

inline const char* placement_policy_to_string(PlacementPolicy policy) {
    switch (policy) {
        case PlacementPolicy::MIXED: return "MIXED";
        case PlacementPolicy::RULE_BASED: return "RULE_BASED";
        case PlacementPolicy::LSTM_SMARTGC: return "LSTM_SMARTGC";
        case PlacementPolicy::SUP_LIKE: return "SUP_LIKE";
        case PlacementPolicy::STAT_ML: return "STAT_ML";
        case PlacementPolicy::LSTM_ATTN_SMARTGC: return "LSTM_ATTN_SMARTGC";
        case PlacementPolicy::HYBRID_ROBUST_SMARTGC: return "HYBRID_ROBUST_SMARTGC";
        default: return "UNKNOWN";
    }
}

inline PlacementPolicy placement_policy_from_string(const std::string& s) {
    if (s == "MIXED") return PlacementPolicy::MIXED;
    if (s == "RULE_BASED") return PlacementPolicy::RULE_BASED;
    if (s == "LSTM_SMARTGC") return PlacementPolicy::LSTM_SMARTGC;
    if (s == "SUP_LIKE") return PlacementPolicy::SUP_LIKE;
    if (s == "STAT_ML") return PlacementPolicy::STAT_ML;
    if (s == "LSTM_ATTN_SMARTGC") return PlacementPolicy::LSTM_ATTN_SMARTGC;
    if (s == "HYBRID_ROBUST_SMARTGC") return PlacementPolicy::HYBRID_ROBUST_SMARTGC;
    throw std::invalid_argument("Unknown placement policy: " + s);
}

// Multi-level GC migration destination classes (Shiro-style), also reused as
// placement-time write streams. For 2-way policies only SHORT (hot) and LONG
// (cold) are used; MEDIUM is exercised by the multi-stream LSTM_ATTN policy.
enum class StreamClass : uint8_t {
    SHORT = 0,   // short remaining lifetime  -> "hot"
    MEDIUM = 1,  // medium remaining lifetime
    LONG = 2     // long remaining lifetime   -> "cold"
};

constexpr size_t MAX_STREAM_CLASSES = 3;

inline const char* stream_class_to_string(StreamClass c) {
    switch (c) {
        case StreamClass::SHORT: return "SHORT";
        case StreamClass::MEDIUM: return "MEDIUM";
        case StreamClass::LONG: return "LONG";
        default: return "UNKNOWN";
    }
}

// Map a predicted 2-way temperature label (1 = HOT, 0 = COLD) to a stream.
inline StreamClass stream_from_binary_class(int predicted_class) {
    return predicted_class == 1 ? StreamClass::SHORT : StreamClass::LONG;
}

// Clamp an arbitrary integer bucket index to a valid StreamClass.
inline StreamClass stream_from_bucket(long bucket, size_t stream_count) {
    if (bucket < 0) bucket = 0;
    size_t b = static_cast<size_t>(bucket);
    if (stream_count == 0) stream_count = 1;
    if (b >= stream_count) b = stream_count - 1;
    if (b > 2) b = 2;
    return static_cast<StreamClass>(b);
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
