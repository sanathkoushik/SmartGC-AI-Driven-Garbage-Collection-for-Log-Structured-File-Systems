#ifndef SMARTGC_CONFIG_HPP
#define SMARTGC_CONFIG_HPP

#include "types.hpp"
#include <cstddef>
#include <cstdint>
#include <string>

namespace smartgc {

struct SimulatorConfig {
    size_t block_size_bytes = 4096;          // 4KB default
    size_t blocks_per_segment = 64;          // 64 blocks per segment (256KB)
    size_t total_segments = 32;              // 32 segments total (8MB storage pool)
    size_t gc_free_segments_threshold = 2;   // Trigger GC when free segments <= threshold
    GcPolicy gc_policy = GcPolicy::GREEDY;
    PlacementPolicy placement_policy = PlacementPolicy::MIXED;
    uint32_t random_seed = 42;

    // ---- Phase 5: multi-level GC migration streams -------------------------
    // Number of destination segment classes GC-reclaimed valid blocks can be
    // routed to (1 = classic single "cold" bucket, up to MAX_STREAM_CLASSES).
    // Also bounds the number of placement-time write log heads.
    size_t migration_stream_count = 1;

    // ---- Phase 5: learned GC trigger (ablation switch) --------------------
    // When true the fixed low-watermark check is replaced by a lightweight
    // contextual controller (Q-table from CSV, or a deterministic adaptive
    // fallback rule). When false the Phase 1 fixed watermark is used verbatim.
    bool learned_trigger = false;
    size_t trigger_state_window_events = 500; // sliding window for burst-rate state
    std::string gc_policy_table_path = "";    // optional Q-table CSV (state,action,q)

    // ---- Phase 4b: confidence gate ---------------------------------------
    // Predictions whose confidence < this threshold fall back to the
    // RULE_BASED decision for that block. <= 0 disables the gate.
    double confidence_fallback_threshold = 0.0;

    size_t total_capacity_blocks() const {
        return total_segments * blocks_per_segment;
    }

    size_t total_capacity_bytes() const {
        return total_capacity_blocks() * block_size_bytes;
    }

    // Effective number of active write log heads for the chosen policy.
    size_t effective_stream_count() const {
        switch (placement_policy) {
            case PlacementPolicy::MIXED:
                return 1;
            case PlacementPolicy::RULE_BASED:
            case PlacementPolicy::SUP_LIKE:
            case PlacementPolicy::STAT_ML:
            case PlacementPolicy::LSTM_SMARTGC:
                return 2; // binary hot/cold
            case PlacementPolicy::LSTM_ATTN_SMARTGC: {
                size_t n = migration_stream_count;
                if (n < 1) n = 1;
                if (n > MAX_STREAM_CLASSES) n = MAX_STREAM_CLASSES;
                return n;
            }
            default:
                return 1;
        }
    }
};

} // namespace smartgc

#endif // SMARTGC_CONFIG_HPP
