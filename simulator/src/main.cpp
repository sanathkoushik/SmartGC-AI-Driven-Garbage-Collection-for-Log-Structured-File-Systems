#include "lfs_simulator.hpp"
#include "synthetic_workload.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <cstring>

using namespace smartgc;

void print_banner() {
    std::cout << "=================================================================\n";
    std::cout << "  SmartGC: Baseline Log-Structured File System (LFS) Simulator   \n";
    std::cout << "=================================================================\n\n";
}

void print_help(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  --total-segments <N>       Total physical segments in storage pool (default: 32)\n"
              << "  --blocks-per-segment <M>   Number of blocks per segment (default: 64)\n"
              << "  --block-size <B>           Size of each block in bytes (default: 4096)\n"
              << "  --gc-threshold <K>         GC trigger free segments threshold (default: 2)\n"
              << "  --requests <R>             Total synthetic write requests to simulate (default: 5000)\n"
              << "  --lba-range <L>            Working set size in LBAs (default: 500)\n"
              << "  --hot-ratio <H>            Fraction of LBAs designated as hot (default: 0.20)\n"
              << "  --hot-traffic <T>          Fraction of traffic directed to hot LBAs (default: 0.80)\n"
              << "  --workload <type>          Workload distribution: 'skewed' or 'uniform' (default: skewed)\n"
              << "  --seed <S>                 Random seed for reproducibility (default: 42)\n"
              << "  --export-metrics <file>    Path to export results CSV row\n"
              << "  --export-trace <file>      Path to export generated normalized trace CSV\n"
              << "  --help                     Display this help message\n\n";
}

int main(int argc, char* argv[]) {
    SimulatorConfig config;
    size_t total_requests = 5000;
    LbaType lba_range = 500;
    double hot_ratio = 0.20;
    double hot_traffic_ratio = 0.80;
    std::string workload_type = "skewed";
    std::string export_metrics_file = "";
    std::string export_trace_file = "";

    // Parse CLI arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_banner();
            print_help(argv[0]);
            return 0;
        } else if (arg == "--total-segments" && i + 1 < argc) {
            config.total_segments = std::stoull(argv[++i]);
        } else if (arg == "--blocks-per-segment" && i + 1 < argc) {
            config.blocks_per_segment = std::stoull(argv[++i]);
        } else if (arg == "--block-size" && i + 1 < argc) {
            config.block_size_bytes = std::stoull(argv[++i]);
        } else if (arg == "--gc-threshold" && i + 1 < argc) {
            config.gc_free_segments_threshold = std::stoull(argv[++i]);
        } else if (arg == "--requests" && i + 1 < argc) {
            total_requests = std::stoull(argv[++i]);
        } else if (arg == "--lba-range" && i + 1 < argc) {
            lba_range = std::stoll(argv[++i]);
        } else if (arg == "--hot-ratio" && i + 1 < argc) {
            hot_ratio = std::stod(argv[++i]);
        } else if (arg == "--hot-traffic" && i + 1 < argc) {
            hot_traffic_ratio = std::stod(argv[++i]);
        } else if (arg == "--workload" && i + 1 < argc) {
            workload_type = argv[++i];
        } else if (arg == "--seed" && i + 1 < argc) {
            config.random_seed = std::stoul(argv[++i]);
        } else if (arg == "--export-metrics" && i + 1 < argc) {
            export_metrics_file = argv[++i];
        } else if (arg == "--export-trace" && i + 1 < argc) {
            export_trace_file = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_help(argv[0]);
            return 1;
        }
    }

    print_banner();

    std::cout << "Configuration:\n";
    std::cout << "  Total Segments:         " << config.total_segments << "\n";
    std::cout << "  Blocks per Segment:     " << config.blocks_per_segment << "\n";
    std::cout << "  Block Size:             " << config.block_size_bytes << " bytes\n";
    std::cout << "  Total Storage Capacity: " << (config.total_capacity_bytes() / 1024.0 / 1024.0) << " MB ("
              << config.total_capacity_blocks() << " blocks)\n";
    std::cout << "  GC Trigger Threshold:   " << config.gc_free_segments_threshold << " free segments\n";
    std::cout << "  Placement Policy:       " << placement_policy_to_string(config.placement_policy) << "\n";
    std::cout << "  GC Policy:              " << gc_policy_to_string(config.gc_policy) << "\n";
    std::cout << "  Random Seed:            " << config.random_seed << "\n\n";

    std::cout << "Generating synthetic workload (" << workload_type << ", " << total_requests << " requests, LBA range [0, " << (lba_range - 1) << "])...\n";

    std::vector<WorkloadRequest> requests;
    if (workload_type == "uniform") {
        requests = SyntheticWorkloadGenerator::generate_uniform(
            total_requests, lba_range, config.random_seed, config.block_size_bytes);
    } else {
        requests = SyntheticWorkloadGenerator::generate_skewed(
            total_requests, lba_range, hot_ratio, hot_traffic_ratio, config.random_seed, config.block_size_bytes);
    }

    if (!export_trace_file.empty()) {
        std::cout << "Exporting normalized trace to: " << export_trace_file << "\n";
        SyntheticWorkloadGenerator::export_to_csv(requests, export_trace_file);
    }

    std::cout << "Executing simulation...\n";
    LfsSimulator sim(config);

    size_t progress_interval = total_requests / 10;
    if (progress_interval == 0) progress_interval = 1;

    for (size_t i = 0; i < requests.size(); ++i) {
        sim.write(requests[i].lba, requests[i].size_bytes);
        if ((i + 1) % progress_interval == 0 || i + 1 == requests.size()) {
            double pct = ((i + 1) * 100.0) / requests.size();
            std::cout << "  Progress: " << std::setw(3) << static_cast<int>(pct) << "% ("
                      << (i + 1) << "/" << requests.size() << " requests, Current WAF: "
                      << std::fixed << std::setprecision(3) << sim.metrics().waf() << ")\r" << std::flush;
        }
    }
    std::cout << "\n\nSimulation completed successfully!\n";

    // Print storage summary and metrics
    sim.print_status_summary();
    std::cout << sim.metrics().to_summary_string() << "\n\n";

    if (!export_metrics_file.empty()) {
        std::cout << "Exporting metrics to: " << export_metrics_file << "\n";
        bool file_exists = false;
        {
            std::ifstream test(export_metrics_file);
            file_exists = test.good();
        }

        std::ofstream out(export_metrics_file, std::ios::app);
        if (out.is_open()) {
            if (!file_exists) {
                out << StorageMetrics::csv_header() << "\n";
            }
            out << sim.metrics().to_csv_row("synthetic_" + workload_type, "MIXED", config.total_segments, config.blocks_per_segment) << "\n";
        }
    }

    return 0;
}
