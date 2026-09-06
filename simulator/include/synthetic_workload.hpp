#ifndef SMARTGC_SYNTHETIC_WORKLOAD_HPP
#define SMARTGC_SYNTHETIC_WORKLOAD_HPP

#include "types.hpp"
#include <vector>
#include <cstdint>
#include <string>

namespace smartgc {

struct WorkloadRequest {
    uint64_t timestamp;
    LbaType lba;
    size_t size_bytes;
    char operation; // 'W', 'R'
};

class SyntheticWorkloadGenerator {
public:
    // Generate Pareto/Zipf-style hot/cold skewed write workload
    static std::vector<WorkloadRequest> generate_skewed(
        size_t total_requests,
        LbaType lba_range_max,
        double hot_ratio = 0.20,
        double hot_traffic_ratio = 0.80,
        uint32_t seed = 42,
        size_t block_size_bytes = 4096);

    // Generate uniform random write workload
    static std::vector<WorkloadRequest> generate_uniform(
        size_t total_requests,
        LbaType lba_range_max,
        uint32_t seed = 42,
        size_t block_size_bytes = 4096);

    // Export generated workload requests to normalized CSV format
    static void export_to_csv(const std::vector<WorkloadRequest>& requests, const std::string& filepath);
};

} // namespace smartgc

#endif // SMARTGC_SYNTHETIC_WORKLOAD_HPP
