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

    // GC trigger low watermark. A user write that needs a fresh segment first
    // cleans until strictly more than this many segments are free.
    size_t gc_free_segments_threshold = 2;

    // Segments that only the garbage collector may consume. This is the
    // over-provisioning reserve that makes a cleaning pass always completable:
    // a victim holds at most blocks_per_segment - 1 valid blocks (a fully valid
    // segment is never selected), so one reserved segment always suffices to
    // absorb a pass, and the victim is returned to the pool immediately after.
    // Must be >= 1.
    size_t gc_reserved_segments = 2;

    // Route GC migrations through their own append point instead of interleaving
    // surviving data into a user log. Kept configurable so that "GC stream
    // separation alone" can be measured as an ablation.
    bool gc_separate_stream = true;

    GcPolicy gc_policy = GcPolicy::GREEDY;
    PlacementPolicy placement_policy = PlacementPolicy::MIXED;
    uint32_t random_seed = 42;

    size_t total_capacity_blocks() const {
        return total_segments * blocks_per_segment;
    }

    size_t total_capacity_bytes() const {
        return total_capacity_blocks() * block_size_bytes;
    }

    // Number of concurrently open user append points implied by the placement
    // policy (1 for MIXED, 2 for the segregated policies).
    size_t user_stream_count() const {
        return placement_policy == PlacementPolicy::MIXED ? 1u : 2u;
    }

    // Segments that can never hold reclaimable user data at any instant: the
    // GC reserve plus the open append points (user streams + the GC stream).
    size_t unavailable_segments() const {
        return gc_reserved_segments + user_stream_count() + (gc_separate_stream ? 1u : 0u);
    }

    // Largest number of simultaneously live (mapped) blocks the pool can hold
    // while still guaranteeing that greedy cleaning can make progress.
    size_t max_live_blocks() const {
        const size_t reserved = unavailable_segments();
        if (total_segments <= reserved) {
            return 0;
        }
        return (total_segments - reserved) * blocks_per_segment;
    }

    // Fraction of raw capacity that may be occupied by live data.
    double max_utilization() const {
        const size_t cap = total_capacity_blocks();
        return cap == 0 ? 0.0 : static_cast<double>(max_live_blocks()) / static_cast<double>(cap);
    }

    // Throws std::invalid_argument describing the first violated constraint.
    void validate() const;

    std::string to_summary_string() const;
};

// Synthetic workload parameters (`synthetic_workload:` section). Synthetic data
// is used for unit tests and development only; conference results are produced
// from real traces.
struct WorkloadConfig {
    size_t total_requests = 5000;
    LbaType lba_range_min = 0;
    LbaType lba_range_max = 500;
    double hot_ratio = 0.20;
    double hot_traffic_ratio = 0.80;

    void validate() const;
};

// Loads the `simulator:` section (and `random_seed`) of a SmartGC config.yaml.
// Only the flat two-level key/value subset that SmartGC actually uses is
// supported; unknown keys inside a recognised section are reported as errors so
// that a typo cannot silently fall back to a default.
SimulatorConfig load_simulator_config(const std::string& yaml_path);

// Loads the `synthetic_workload:` section of a SmartGC config.yaml.
WorkloadConfig load_workload_config(const std::string& yaml_path);

} // namespace smartgc

#endif // SMARTGC_CONFIG_HPP
