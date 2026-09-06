#ifndef SMARTGC_CONFIG_HPP
#define SMARTGC_CONFIG_HPP

#include "types.hpp"
#include <cstddef>
#include <cstdint>

namespace smartgc {

struct SimulatorConfig {
    size_t block_size_bytes = 4096;          // 4KB default
    size_t blocks_per_segment = 64;          // 64 blocks per segment (256KB)
    size_t total_segments = 32;              // 32 segments total (8MB storage pool)
    size_t gc_free_segments_threshold = 2;   // Trigger GC when free segments <= threshold
    GcPolicy gc_policy = GcPolicy::GREEDY;
    PlacementPolicy placement_policy = PlacementPolicy::MIXED;
    uint32_t random_seed = 42;

    size_t total_capacity_blocks() const {
        return total_segments * blocks_per_segment;
    }

    size_t total_capacity_bytes() const {
        return total_capacity_blocks() * block_size_bytes;
    }
};

} // namespace smartgc

#endif // SMARTGC_CONFIG_HPP
