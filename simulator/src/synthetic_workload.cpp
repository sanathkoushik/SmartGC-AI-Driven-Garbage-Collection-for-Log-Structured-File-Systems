#include "synthetic_workload.hpp"
#include <random>
#include <fstream>
#include <stdexcept>
#include <algorithm>

namespace smartgc {

std::vector<WorkloadRequest> SyntheticWorkloadGenerator::generate_skewed(
    size_t total_requests,
    LbaType lba_range_max,
    double hot_ratio,
    double hot_traffic_ratio,
    uint32_t seed) {

    if (lba_range_max <= 0) {
        throw std::invalid_argument("lba_range_max must be > 0");
    }
    if (hot_ratio <= 0.0 || hot_ratio >= 1.0) {
        throw std::invalid_argument("hot_ratio must be in (0, 1)");
    }
    if (hot_traffic_ratio < 0.0 || hot_traffic_ratio > 1.0) {
        throw std::invalid_argument("hot_traffic_ratio must be in [0, 1]");
    }

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);

    const LbaType hot_lba_boundary =
        std::max<LbaType>(1, static_cast<LbaType>(static_cast<double>(lba_range_max) * hot_ratio));
    std::uniform_int_distribution<LbaType> hot_dist(0, hot_lba_boundary - 1);
    std::uniform_int_distribution<LbaType> cold_dist(hot_lba_boundary, lba_range_max - 1);

    std::vector<WorkloadRequest> requests;
    requests.reserve(total_requests);

    for (size_t i = 0; i < total_requests; ++i) {
        LbaType selected_lba;
        if (prob_dist(rng) < hot_traffic_ratio || hot_lba_boundary >= lba_range_max) {
            selected_lba = hot_dist(rng);
        } else {
            selected_lba = cold_dist(rng);
        }

        requests.push_back(WorkloadRequest{static_cast<uint64_t>(i), selected_lba, 1u, 'W'});
    }

    return requests;
}

std::vector<WorkloadRequest> SyntheticWorkloadGenerator::generate_uniform(
    size_t total_requests,
    LbaType lba_range_max,
    uint32_t seed) {

    if (lba_range_max <= 0) {
        throw std::invalid_argument("lba_range_max must be > 0");
    }

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<LbaType> lba_dist(0, lba_range_max - 1);

    std::vector<WorkloadRequest> requests;
    requests.reserve(total_requests);

    for (size_t i = 0; i < total_requests; ++i) {
        requests.push_back(WorkloadRequest{static_cast<uint64_t>(i), lba_dist(rng), 1u, 'W'});
    }

    return requests;
}

void SyntheticWorkloadGenerator::export_to_csv(
    const std::vector<WorkloadRequest>& requests,
    const std::string& filepath,
    const std::string& trace_id) {

    std::ofstream out(filepath);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }

    out << "timestamp,lba,size,operation,trace_id\n";
    for (const WorkloadRequest& req : requests) {
        out << req.timestamp << ','
            << req.lba << ','
            << req.size_blocks << ','
            << req.operation << ','
            << trace_id << '\n';
    }
}

} // namespace smartgc
