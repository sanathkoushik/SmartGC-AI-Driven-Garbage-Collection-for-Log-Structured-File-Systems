#include "lfs_simulator.hpp"
#include "synthetic_workload.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cmath>

using namespace smartgc;

namespace {

void print_banner() {
    std::cout << "=================================================================\n";
    std::cout << "  SmartGC: Log-Structured File System (LFS) Simulator            \n";
    std::cout << "=================================================================\n\n";
}

void print_help(const char* prog) {
    std::cout
        << "Usage: " << prog << " [options]\n\n"
        << "Configuration:\n"
        << "  --config <path>            Load simulator + workload settings from a config.yaml\n"
        << "                             (CLI options given after --config override it)\n"
        << "Geometry:\n"
        << "  --total-segments <N>       Total physical segments in the storage pool\n"
        << "  --blocks-per-segment <M>   Blocks per segment\n"
        << "  --block-size <B>           Block size in bytes\n"
        << "  --gc-threshold <K>         Clean when free segments <= K\n"
        << "  --gc-reserved <R>          Segments reserved for GC only (>= 1)\n"
        << "  --gc-shared-stream         Interleave GC migrations into the main user log\n"
        << "  --placement <P>            MIXED | RULE_BASED | LSTM_SMARTGC\n"
        << "Workload (synthetic; development and tests only):\n"
        << "  --requests <R>             Total synthetic write requests\n"
        << "  --lba-range <L>            Working-set size in distinct LBAs\n"
        << "  --utilization <U>          Derive --lba-range as U x raw capacity (0 < U <= 1);\n"
        << "                             overrides --lba-range\n"
        << "  --hot-ratio <H>            Fraction of LBAs designated hot\n"
        << "  --hot-traffic <T>          Fraction of traffic directed at hot LBAs\n"
        << "  --workload <type>          'skewed' or 'uniform'\n"
        << "  --seed <S>                 Random seed\n"
        << "Output:\n"
        << "  --export-metrics <file>    Append a metrics CSV row (header written if new)\n"
        << "  --export-trace <file>      Write the generated normalized trace CSV\n"
        << "  --label <name>             Trace label recorded in the metrics row\n"
        << "  --quiet                    Suppress the per-segment status dump\n"
        << "  --help                     Show this message\n\n";
}

bool needs_value(int i, int argc, const std::string& arg) {
    if (i + 1 >= argc) {
        std::cerr << "Error: option " << arg << " requires a value\n";
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char* argv[]) {
    SimulatorConfig config;
    WorkloadConfig workload;

    std::string workload_type = "skewed";
    std::string export_metrics_file;
    std::string export_trace_file;
    std::string label;
    double utilization_target = 0.0;
    bool quiet = false;
    bool lba_range_explicit = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        try {
            if (arg == "--help" || arg == "-h") {
                print_banner();
                print_help(argv[0]);
                return 0;
            } else if (arg == "--config") {
                if (!needs_value(i, argc, arg)) return 2;
                const std::string path = argv[++i];
                config = load_simulator_config(path);
                workload = load_workload_config(path);
            } else if (arg == "--total-segments") {
                if (!needs_value(i, argc, arg)) return 2;
                config.total_segments = std::stoull(argv[++i]);
            } else if (arg == "--blocks-per-segment") {
                if (!needs_value(i, argc, arg)) return 2;
                config.blocks_per_segment = std::stoull(argv[++i]);
            } else if (arg == "--block-size") {
                if (!needs_value(i, argc, arg)) return 2;
                config.block_size_bytes = std::stoull(argv[++i]);
            } else if (arg == "--gc-threshold") {
                if (!needs_value(i, argc, arg)) return 2;
                config.gc_free_segments_threshold = std::stoull(argv[++i]);
            } else if (arg == "--gc-reserved") {
                if (!needs_value(i, argc, arg)) return 2;
                config.gc_reserved_segments = std::stoull(argv[++i]);
            } else if (arg == "--gc-shared-stream") {
                config.gc_separate_stream = false;
            } else if (arg == "--placement") {
                if (!needs_value(i, argc, arg)) return 2;
                const std::string value = argv[++i];
                if (!parse_placement_policy(value, config.placement_policy)) {
                    std::cerr << "Error: --placement expects MIXED, RULE_BASED or LSTM_SMARTGC, got '"
                              << value << "'\n";
                    return 2;
                }
            } else if (arg == "--requests") {
                if (!needs_value(i, argc, arg)) return 2;
                workload.total_requests = std::stoull(argv[++i]);
            } else if (arg == "--lba-range") {
                if (!needs_value(i, argc, arg)) return 2;
                workload.lba_range_max = std::stoll(argv[++i]);
                lba_range_explicit = true;
            } else if (arg == "--utilization") {
                if (!needs_value(i, argc, arg)) return 2;
                utilization_target = std::stod(argv[++i]);
            } else if (arg == "--hot-ratio") {
                if (!needs_value(i, argc, arg)) return 2;
                workload.hot_ratio = std::stod(argv[++i]);
            } else if (arg == "--hot-traffic") {
                if (!needs_value(i, argc, arg)) return 2;
                workload.hot_traffic_ratio = std::stod(argv[++i]);
            } else if (arg == "--workload") {
                if (!needs_value(i, argc, arg)) return 2;
                workload_type = argv[++i];
                if (workload_type != "skewed" && workload_type != "uniform") {
                    std::cerr << "Error: --workload expects 'skewed' or 'uniform'\n";
                    return 2;
                }
            } else if (arg == "--seed") {
                if (!needs_value(i, argc, arg)) return 2;
                config.random_seed = static_cast<uint32_t>(std::stoul(argv[++i]));
            } else if (arg == "--export-metrics") {
                if (!needs_value(i, argc, arg)) return 2;
                export_metrics_file = argv[++i];
            } else if (arg == "--export-trace") {
                if (!needs_value(i, argc, arg)) return 2;
                export_trace_file = argv[++i];
            } else if (arg == "--label") {
                if (!needs_value(i, argc, arg)) return 2;
                label = argv[++i];
            } else if (arg == "--quiet") {
                quiet = true;
            } else {
                std::cerr << "Unknown argument: " << arg << "\n\n";
                print_help(argv[0]);
                return 2;
            }
        } catch (const std::exception& ex) {
            std::cerr << "Error processing " << arg << ": " << ex.what() << "\n";
            return 2;
        }
    }

    try {
        config.validate();

        if (utilization_target > 0.0) {
            if (utilization_target > 1.0) {
                std::cerr << "Error: --utilization must be in (0, 1]\n";
                return 2;
            }
            const double target_blocks =
                utilization_target * static_cast<double>(config.total_capacity_blocks());
            workload.lba_range_max = static_cast<LbaType>(std::llround(target_blocks));
            if (workload.lba_range_max < 1) {
                workload.lba_range_max = 1;
            }
        } else if (!lba_range_explicit && workload.lba_range_max <= 0) {
            std::cerr << "Error: no working-set size configured\n";
            return 2;
        }
        workload.validate();

        const size_t working_set = static_cast<size_t>(workload.lba_range_max);
        if (working_set > config.max_live_blocks()) {
            std::cerr << "Error: working set of " << working_set
                      << " blocks exceeds the usable capacity of " << config.max_live_blocks()
                      << " blocks (" << std::fixed << std::setprecision(2)
                      << (config.max_utilization() * 100.0) << "% of raw capacity).\n"
                      << "       Add segments, or lower --gc-reserved (currently "
                      << config.gc_reserved_segments << ").\n";
            return 3;
        }

        const double actual_utilization =
            static_cast<double>(working_set) / static_cast<double>(config.total_capacity_blocks());

        print_banner();
        std::cout << "Configuration:\n" << config.to_summary_string();
        std::cout << "  Working Set:            " << working_set << " blocks ("
                  << std::fixed << std::setprecision(2) << (actual_utilization * 100.0)
                  << "% of raw capacity)\n\n";

        std::cout << "Generating synthetic workload (" << workload_type << ", "
                  << workload.total_requests << " requests, LBA range [0, "
                  << (workload.lba_range_max - 1) << "])...\n";
        std::cout << "NOTE: synthetic data is for development and tests only; "
                     "conference results use real traces.\n";

        std::vector<WorkloadRequest> requests =
            workload_type == "uniform"
                ? SyntheticWorkloadGenerator::generate_uniform(
                      workload.total_requests, workload.lba_range_max, config.random_seed)
                : SyntheticWorkloadGenerator::generate_skewed(
                      workload.total_requests, workload.lba_range_max, workload.hot_ratio,
                      workload.hot_traffic_ratio, config.random_seed);

        const std::string trace_label =
            label.empty() ? ("synthetic_" + workload_type) : label;

        if (!export_trace_file.empty()) {
            std::cout << "Exporting normalized trace to: " << export_trace_file << "\n";
            SyntheticWorkloadGenerator::export_to_csv(requests, export_trace_file, trace_label);
        }

        std::cout << "Executing simulation...\n";
        LfsSimulator sim(config);

        size_t progress_interval = requests.size() / 10;
        if (progress_interval == 0) progress_interval = 1;

        for (size_t i = 0; i < requests.size(); ++i) {
            sim.write(requests[i].lba,
                      static_cast<size_t>(requests[i].size_blocks) * config.block_size_bytes);
            if ((i + 1) % progress_interval == 0 || i + 1 == requests.size()) {
                const double pct = static_cast<double>(i + 1) * 100.0 / static_cast<double>(requests.size());
                std::cout << "  Progress: " << std::setw(3) << static_cast<int>(pct) << "% ("
                          << (i + 1) << "/" << requests.size() << " requests, current WAF: "
                          << std::fixed << std::setprecision(3) << sim.metrics().waf() << ")\r"
                          << std::flush;
            }
        }
        std::cout << "\n\nSimulation completed successfully.\n";

        if (!quiet) {
            sim.print_status_summary();
        }
        std::cout << sim.metrics().to_summary_string() << "\n\n";

        if (!export_metrics_file.empty()) {
            std::cout << "Exporting metrics to: " << export_metrics_file << "\n";
            bool file_exists = false;
            {
                std::ifstream test(export_metrics_file);
                file_exists = test.good();
            }
            std::ofstream out(export_metrics_file, std::ios::app);
            if (!out.is_open()) {
                std::cerr << "Error: cannot open metrics file for writing: "
                          << export_metrics_file << "\n";
                return 4;
            }
            if (!file_exists) {
                out << StorageMetrics::csv_header() << "\n";
            }

            RunDescriptor run;
            run.dataset = "synthetic";
            run.trace = trace_label;
            run.policy = placement_policy_to_string(config.placement_policy);
            run.model_type = "none";
            run.total_segments = config.total_segments;
            run.blocks_per_segment = config.blocks_per_segment;
            run.block_size_bytes = config.block_size_bytes;
            run.gc_reserved_segments = config.gc_reserved_segments;
            run.gc_separate_stream = config.gc_separate_stream;
            run.target_utilization = actual_utilization;
            run.seed = config.random_seed;
            out << run.to_csv_row(sim.metrics()) << "\n";
        }

        return 0;

    } catch (const DeviceFullError& ex) {
        std::cerr << "\nDEVICE FULL: " << ex.what() << "\n";
        return 3;
    } catch (const std::exception& ex) {
        std::cerr << "\nERROR: " << ex.what() << "\n";
        return 1;
    }
}
