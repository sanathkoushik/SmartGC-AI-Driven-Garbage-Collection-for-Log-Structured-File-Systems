#ifndef SMARTGC_SYNTHETIC_WORKLOAD_HPP
#define SMARTGC_SYNTHETIC_WORKLOAD_HPP

#include "types.hpp"
#include <vector>
#include <cstdint>
#include <string>

namespace smartgc {

// One record of the normalized trace contract (docs/architecture.md 2.1).
// `size_blocks` is a count of block-size units, never bytes: multi-block
// requests are expanded during preprocessing so that every record is one block.
struct WorkloadRequest {
    uint64_t timestamp;
    LbaType lba;
    uint32_t size_blocks;
    char operation;      // 'W', 'R' or 'D'
};

// Synthetic workloads exist for unit tests and development only. Conference
// results are produced exclusively from real block-I/O traces (docs/dataset.md).
class SyntheticWorkloadGenerator {
public:
    // Pareto/Zipf-style hot/cold skewed write workload.
    static std::vector<WorkloadRequest> generate_skewed(
        size_t total_requests,
        LbaType lba_range_max,
        double hot_ratio = 0.20,
        double hot_traffic_ratio = 0.80,
        uint32_t seed = 42);

    // Uniform random write workload.
    static std::vector<WorkloadRequest> generate_uniform(
        size_t total_requests,
        LbaType lba_range_max,
        uint32_t seed = 42);

    // Writes the normalized trace contract: timestamp,lba,size,operation,trace_id
    static void export_to_csv(const std::vector<WorkloadRequest>& requests,
                              const std::string& filepath,
                              const std::string& trace_id);
};

} // namespace smartgc

#endif // SMARTGC_SYNTHETIC_WORKLOAD_HPP
