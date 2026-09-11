#include "lfs_simulator.hpp"
#include "synthetic_workload.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

using namespace smartgc;

void print_banner() {
    std::cout << "=================================================================\n";
    std::cout << "  SmartGC: Log-Structured File System (LFS) Simulator            \n";
    std::cout << "  Placement ladder + multi-stream GC migration + learned trigger \n";
    std::cout << "=================================================================\n\n";
}

void print_help(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n\n"
              << "Storage geometry:\n"
              << "  --total-segments <N>        Total physical segments (default: 32)\n"
              << "  --blocks-per-segment <M>    Blocks per segment (default: 64)\n"
              << "  --block-size <B>            Block size in bytes (default: 4096)\n"
              << "  --gc-threshold <K>          Fixed GC low-watermark, free segments (default: 2)\n\n"
              << "Policy:\n"
              << "  --placement-policy <P>      MIXED | RULE_BASED | SUP_LIKE | STAT_ML |\n"
              << "                              LSTM_SMARTGC | LSTM_ATTN_SMARTGC |\n"
              << "                              HYBRID_ROBUST_SMARTGC (default: MIXED)\n"
              << "  --migration-streams <N>     GC migration destination classes 1..3 (default: 1)\n"
              << "  --learned-trigger          Enable the learned/adaptive GC trigger\n"
              << "  --gc-policy-table <file>    Q-table CSV for the learned trigger\n"
              << "  --trigger-window <N>        Trigger state window in write events (default: 500)\n"
              << "  --confidence-threshold <x>  Fall back to RULE_BASED when confidence < x (default: 0)\n\n"
              << "Input:\n"
              << "  --trace <file>             Replay a normalized trace CSV (timestamp,lba,size,operation)\n"
              << "  --predictions <file>       predictions.csv consumed in lockstep with writes\n"
              << "  --requests <R>             Synthetic write requests when no --trace (default: 5000)\n"
              << "  --lba-range <L>            Synthetic working set size in LBAs (default: 500)\n"
              << "  --hot-ratio <H>            Synthetic hot LBA fraction (default: 0.20)\n"
              << "  --hot-traffic <T>          Synthetic hot traffic fraction (default: 0.80)\n"
              << "  --workload <type>          'skewed' or 'uniform' (default: skewed)\n"
              << "  --seed <S>                 Random seed (default: 42)\n\n"
              << "Output:\n"
              << "  --workload-name <str>      Name written to the metrics row\n"
              << "  --run-id <str>             Reproducibility hash written to the metrics row\n"
              << "  --export-metrics <file>    Append a metrics CSV row (contract v2)\n"
              << "  --export-trace <file>      Export the generated synthetic trace CSV\n"
              << "  --quiet                    Suppress per-segment status dump\n"
              << "  --help                     Show this help\n\n";
}

struct TraceEvent {
    LbaType lba;
    size_t size_bytes;
    char op;
};

static std::vector<TraceEvent> load_normalized_trace(const std::string& path, size_t block_size) {
    std::vector<TraceEvent> events;
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open trace file: " + path);
    }
    std::string line;
    std::getline(in, line); // header
    // locate columns
    std::vector<std::string> header;
    {
        std::istringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            std::transform(tok.begin(), tok.end(), tok.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            // trim
            size_t b = tok.find_first_not_of(" \t\r\n");
            size_t e = tok.find_last_not_of(" \t\r\n");
            header.push_back(b == std::string::npos ? std::string() : tok.substr(b, e - b + 1));
        }
    }
    auto col = [&](const std::string& name) -> int {
        for (size_t i = 0; i < header.size(); ++i) if (header[i] == name) return static_cast<int>(i);
        return -1;
    };
    const int c_lba = col("lba");
    const int c_size = col("size");
    const int c_op = col("operation");
    if (c_lba < 0) throw std::runtime_error("Trace missing 'lba' column: " + path);

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<std::string> f;
        std::istringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) f.push_back(tok);
        if (static_cast<int>(f.size()) <= c_lba) continue;

        TraceEvent ev;
        ev.lba = static_cast<LbaType>(std::stoll(f[c_lba]));
        ev.size_bytes = block_size;
        if (c_size >= 0 && c_size < static_cast<int>(f.size()) && !f[c_size].empty()) {
            const long s = std::stol(f[c_size]);
            ev.size_bytes = s > 0 ? static_cast<size_t>(s) * block_size : block_size;
        }
        ev.op = 'W';
        if (c_op >= 0 && c_op < static_cast<int>(f.size()) && !f[c_op].empty()) {
            ev.op = static_cast<char>(std::toupper(static_cast<unsigned char>(f[c_op][0])));
        }
        events.push_back(ev);
    }
    return events;
}

int main(int argc, char* argv[]) {
    SimulatorConfig config;
    size_t total_requests = 5000;
    LbaType lba_range = 500;
    double hot_ratio = 0.20;
    double hot_traffic_ratio = 0.80;
    std::string workload_type = "skewed";
    std::string export_metrics_file;
    std::string export_trace_file;
    std::string trace_file;
    std::string predictions_file;
    std::string workload_name;
    std::string run_id;
    bool quiet = false;
    bool generate_only = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? std::string(argv[++i]) : std::string(); };
        if (arg == "--help" || arg == "-h") {
            print_banner(); print_help(argv[0]); return 0;
        } else if (arg == "--total-segments") { config.total_segments = std::stoull(next());
        } else if (arg == "--blocks-per-segment") { config.blocks_per_segment = std::stoull(next());
        } else if (arg == "--block-size") { config.block_size_bytes = std::stoull(next());
        } else if (arg == "--gc-threshold") { config.gc_free_segments_threshold = std::stoull(next());
        } else if (arg == "--placement-policy") { config.placement_policy = placement_policy_from_string(next());
        } else if (arg == "--migration-streams") { config.migration_stream_count = std::stoull(next());
        } else if (arg == "--learned-trigger") { config.learned_trigger = true;
        } else if (arg == "--gc-policy-table") { config.gc_policy_table_path = next();
        } else if (arg == "--trigger-window") { config.trigger_state_window_events = std::stoull(next());
        } else if (arg == "--confidence-threshold") { config.confidence_fallback_threshold = std::stod(next());
        } else if (arg == "--trace") { trace_file = next();
        } else if (arg == "--predictions") { predictions_file = next();
        } else if (arg == "--requests") { total_requests = std::stoull(next());
        } else if (arg == "--lba-range") { lba_range = std::stoll(next());
        } else if (arg == "--hot-ratio") { hot_ratio = std::stod(next());
        } else if (arg == "--hot-traffic") { hot_traffic_ratio = std::stod(next());
        } else if (arg == "--workload") { workload_type = next();
        } else if (arg == "--seed") { config.random_seed = std::stoul(next());
        } else if (arg == "--workload-name") { workload_name = next();
        } else if (arg == "--run-id") { run_id = next();
        } else if (arg == "--export-metrics") { export_metrics_file = next();
        } else if (arg == "--export-trace") { export_trace_file = next();
        } else if (arg == "--quiet") { quiet = true;
        } else if (arg == "--generate-only") { generate_only = true;
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
    std::cout << "  GC Trigger:             " << (config.learned_trigger ? "LEARNED/adaptive" : "fixed watermark")
              << " (watermark " << config.gc_free_segments_threshold << ")\n";
    std::cout << "  Placement Policy:       " << placement_policy_to_string(config.placement_policy) << "\n";
    std::cout << "  Migration Streams:      " << config.effective_stream_count() << "\n";
    std::cout << "  Contract Version:       " << CONTRACT_VERSION << "\n";
    std::cout << "  Random Seed:            " << config.random_seed << "\n\n";

    // Build the write stream (trace replay or synthetic generation).
    std::vector<LbaType> write_lbas;
    std::vector<size_t> write_sizes;
    std::string resolved_workload_name = workload_name;

    if (!trace_file.empty()) {
        std::cout << "Replaying normalized trace: " << trace_file << "\n";
        const std::vector<TraceEvent> events = load_normalized_trace(trace_file, config.block_size_bytes);
        for (const auto& ev : events) {
            if (ev.op == 'W' || ev.op == 'D') {
                write_lbas.push_back(ev.lba);
                write_sizes.push_back(ev.size_bytes);
            }
        }
        if (resolved_workload_name.empty()) {
            size_t slash = trace_file.find_last_of("/\\");
            resolved_workload_name = (slash == std::string::npos) ? trace_file : trace_file.substr(slash + 1);
        }
        std::cout << "  Loaded " << events.size() << " events, " << write_lbas.size() << " writes.\n";
    } else {
        std::cout << "Generating synthetic workload (" << workload_type << ", " << total_requests
                  << " requests, LBA range [0, " << (lba_range - 1) << "])...\n";
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
        if (generate_only) {
            std::cout << "--generate-only: wrote " << requests.size() << " requests, skipping simulation.\n";
            return 0;
        }
        for (const auto& r : requests) {
            write_lbas.push_back(r.lba);
            write_sizes.push_back(r.size_bytes);
        }
        if (resolved_workload_name.empty()) resolved_workload_name = "synthetic_" + workload_type;
    }

    std::cout << "Executing simulation...\n";
    LfsSimulator sim(config);
    if (!predictions_file.empty()) {
        std::cout << "Loading predictions: " << predictions_file << "\n";
        sim.load_predictions(predictions_file);
    }

    size_t progress_interval = write_lbas.size() / 10;
    if (progress_interval == 0) progress_interval = 1;

    for (size_t i = 0; i < write_lbas.size(); ++i) {
        sim.write(write_lbas[i], write_sizes[i]);
        if ((i + 1) % progress_interval == 0 || i + 1 == write_lbas.size()) {
            double pct = ((i + 1) * 100.0) / write_lbas.size();
            std::cout << "  Progress: " << std::setw(3) << static_cast<int>(pct) << "% ("
                      << (i + 1) << "/" << write_lbas.size() << " writes, Current WAF: "
                      << std::fixed << std::setprecision(3) << sim.metrics().waf() << ")\r" << std::flush;
        }
    }
    std::cout << "\n\nSimulation completed successfully!\n";

    if (!quiet) sim.print_status_summary();
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
            if (!file_exists) out << StorageMetrics::csv_header() << "\n";
            RunContext ctx;
            ctx.workload_name = resolved_workload_name;
            ctx.placement_policy = placement_policy_to_string(config.placement_policy);
            ctx.run_id = run_id;
            ctx.total_segments = config.total_segments;
            ctx.blocks_per_segment = config.blocks_per_segment;
            ctx.migration_stream_count = sim.stream_count();
            ctx.learned_trigger_enabled = config.learned_trigger;
            ctx.avg_gc_decision_latency_us = sim.avg_gc_decision_latency_us();
            out << sim.metrics().to_csv_row(ctx) << "\n";
        }
    }

    return 0;
}
