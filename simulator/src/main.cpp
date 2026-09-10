#include "lfs_simulator.hpp"
#include "prediction_reader.hpp"
#include "rule_based_classifier.hpp"
#include "synthetic_workload.hpp"
#include "trace_reader.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace smartgc;

namespace {

struct RunOptions {
    std::string trace_path;
    std::string predictions_path;
    std::string export_metrics;
    std::string export_trace;
    std::string report_json;
    std::string dataset = "synthetic";
    std::string trace_name;
    std::string model_type = "none";
    std::string workload_type = "skewed";
    double rule_threshold_us = -1.0;
    size_t rule_min_history = 10;
    uint64_t max_writes = 0;          // 0 = unlimited
    double utilization_target = 0.0;
    bool quiet = false;
    bool lba_range_explicit = false;
};

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
        << "Geometry:\n"
        << "  --total-segments <N>       Total physical segments in the storage pool\n"
        << "  --blocks-per-segment <M>   Blocks per segment\n"
        << "  --block-size <B>           Block size in bytes\n"
        << "  --gc-threshold <K>         Clean when free segments <= K\n"
        << "  --gc-reserved <R>          Segments reserved for GC only (>= 1)\n"
        << "  --gc-shared-stream         Interleave GC migrations into the main user log\n"
        << "  --placement <P>            MIXED | RULE_BASED | LSTM_SMARTGC\n"
        << "Real trace replay:\n"
        << "  --trace <file>             Normalized trace CSV (timestamp,lba,size,operation,trace_id)\n"
        << "  --predictions <file>       Prediction CSV, required for LSTM_SMARTGC\n"
        << "  --rule-threshold-us <T>    RULE_BASED hot cutoff in microseconds (from TRAINING data)\n"
        << "  --rule-min-history <N>     Intervals a LBA needs before the rule will classify it\n"
        << "  --max-writes <N>           Stop after N write requests (0 = whole trace)\n"
        << "Synthetic workload (development and tests only):\n"
        << "  --requests <R>             Total synthetic write requests\n"
        << "  --lba-range <L>            Working-set size in distinct LBAs\n"
        << "  --utilization <U>          Derive --lba-range as U x raw capacity (0 < U <= 1)\n"
        << "  --hot-ratio <H>            Fraction of LBAs designated hot\n"
        << "  --hot-traffic <T>          Fraction of traffic directed at hot LBAs\n"
        << "  --workload <type>          'skewed' or 'uniform'\n"
        << "  --seed <S>                 Random seed\n"
        << "Output:\n"
        << "  --export-metrics <file>    Append a metrics CSV row (header written if new)\n"
        << "  --export-trace <file>      Write the generated synthetic trace CSV\n"
        << "  --report-json <file>       Write full run diagnostics as JSON\n"
        << "  --dataset <name>           Dataset label recorded in the metrics row\n"
        << "  --trace-name <name>        Trace label recorded in the metrics row\n"
        << "  --model-type <name>        Model label recorded in the metrics row\n"
        << "  --quiet                    Suppress the per-segment status dump\n"
        << "  --help                     Show this message\n\n"
        << "Exit codes: 0 ok, 1 internal error, 2 bad arguments, 3 device full.\n\n";
}

bool needs_value(int i, int argc, const std::string& arg) {
    if (i + 1 >= argc) {
        std::cerr << "Error: option " << arg << " requires a value\n";
        return false;
    }
    return true;
}

std::string json_escape(const std::string& value) {
    std::string out;
    for (const char c : value) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (c == '\n') {
            out += "\\n";
        } else {
            out += c;
        }
    }
    return out;
}

// Replays a normalized trace. Reads are counted but do not change device state:
// in a log-structured store only writes allocate blocks and drive cleaning.
struct ReplayCounters {
    uint64_t write_requests = 0;
    uint64_t read_requests = 0;
    uint64_t discard_requests = 0;
    uint64_t hot_placements = 0;
    uint64_t cold_placements = 0;
    uint64_t unknown_placements = 0;
};

} // namespace

int main(int argc, char* argv[]) {
    SimulatorConfig config;
    WorkloadConfig workload;
    RunOptions options;

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
            } else if (arg == "--trace") {
                if (!needs_value(i, argc, arg)) return 2;
                options.trace_path = argv[++i];
            } else if (arg == "--predictions") {
                if (!needs_value(i, argc, arg)) return 2;
                options.predictions_path = argv[++i];
            } else if (arg == "--rule-threshold-us") {
                if (!needs_value(i, argc, arg)) return 2;
                options.rule_threshold_us = std::stod(argv[++i]);
            } else if (arg == "--rule-min-history") {
                if (!needs_value(i, argc, arg)) return 2;
                options.rule_min_history = std::stoull(argv[++i]);
            } else if (arg == "--max-writes") {
                if (!needs_value(i, argc, arg)) return 2;
                options.max_writes = std::stoull(argv[++i]);
            } else if (arg == "--requests") {
                if (!needs_value(i, argc, arg)) return 2;
                workload.total_requests = std::stoull(argv[++i]);
            } else if (arg == "--lba-range") {
                if (!needs_value(i, argc, arg)) return 2;
                workload.lba_range_max = std::stoll(argv[++i]);
                options.lba_range_explicit = true;
            } else if (arg == "--utilization") {
                if (!needs_value(i, argc, arg)) return 2;
                options.utilization_target = std::stod(argv[++i]);
            } else if (arg == "--hot-ratio") {
                if (!needs_value(i, argc, arg)) return 2;
                workload.hot_ratio = std::stod(argv[++i]);
            } else if (arg == "--hot-traffic") {
                if (!needs_value(i, argc, arg)) return 2;
                workload.hot_traffic_ratio = std::stod(argv[++i]);
            } else if (arg == "--workload") {
                if (!needs_value(i, argc, arg)) return 2;
                options.workload_type = argv[++i];
                if (options.workload_type != "skewed" && options.workload_type != "uniform") {
                    std::cerr << "Error: --workload expects 'skewed' or 'uniform'\n";
                    return 2;
                }
            } else if (arg == "--seed") {
                if (!needs_value(i, argc, arg)) return 2;
                config.random_seed = static_cast<uint32_t>(std::stoul(argv[++i]));
            } else if (arg == "--export-metrics") {
                if (!needs_value(i, argc, arg)) return 2;
                options.export_metrics = argv[++i];
            } else if (arg == "--export-trace") {
                if (!needs_value(i, argc, arg)) return 2;
                options.export_trace = argv[++i];
            } else if (arg == "--report-json") {
                if (!needs_value(i, argc, arg)) return 2;
                options.report_json = argv[++i];
            } else if (arg == "--dataset") {
                if (!needs_value(i, argc, arg)) return 2;
                options.dataset = argv[++i];
            } else if (arg == "--trace-name") {
                if (!needs_value(i, argc, arg)) return 2;
                options.trace_name = argv[++i];
            } else if (arg == "--model-type") {
                if (!needs_value(i, argc, arg)) return 2;
                options.model_type = argv[++i];
            } else if (arg == "--quiet") {
                options.quiet = true;
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

        const bool replay_trace = !options.trace_path.empty();

        // A prediction-driven policy without predictions would silently degrade
        // into MIXED and quietly invalidate the comparison.
        if (config.placement_policy == PlacementPolicy::LSTM_SMARTGC
            && options.predictions_path.empty()) {
            std::cerr << "Error: --placement LSTM_SMARTGC requires --predictions <file>\n";
            return 2;
        }
        if (config.placement_policy == PlacementPolicy::RULE_BASED
            && options.rule_threshold_us <= 0.0) {
            std::cerr << "Error: --placement RULE_BASED requires --rule-threshold-us <T>,\n"
                      << "       derived from TRAINING data (see docs/methodology.md).\n";
            return 2;
        }
        if (!replay_trace && !options.predictions_path.empty()) {
            std::cerr << "Error: --predictions only applies to a real trace replay (--trace)\n";
            return 2;
        }

        if (!options.quiet) {
            print_banner();
        }

        double actual_utilization = 0.0;
        std::vector<WorkloadRequest> synthetic_requests;
        std::string trace_label = options.trace_name;

        if (!replay_trace) {
            if (options.utilization_target > 0.0) {
                if (options.utilization_target > 1.0) {
                    std::cerr << "Error: --utilization must be in (0, 1]\n";
                    return 2;
                }
                workload.lba_range_max = static_cast<LbaType>(std::llround(
                    options.utilization_target * static_cast<double>(config.total_capacity_blocks())));
                if (workload.lba_range_max < 1) workload.lba_range_max = 1;
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
            actual_utilization = static_cast<double>(working_set)
                                 / static_cast<double>(config.total_capacity_blocks());
            if (trace_label.empty()) trace_label = "synthetic_" + options.workload_type;

            synthetic_requests = options.workload_type == "uniform"
                ? SyntheticWorkloadGenerator::generate_uniform(
                      workload.total_requests, workload.lba_range_max, config.random_seed)
                : SyntheticWorkloadGenerator::generate_skewed(
                      workload.total_requests, workload.lba_range_max, workload.hot_ratio,
                      workload.hot_traffic_ratio, config.random_seed);

            if (!options.export_trace.empty()) {
                SyntheticWorkloadGenerator::export_to_csv(synthetic_requests,
                                                          options.export_trace, trace_label);
            }
        }

        if (!options.quiet) {
            std::cout << "Configuration:\n" << config.to_summary_string();
            if (replay_trace) {
                std::cout << "  Trace:                  " << options.trace_path << "\n";
                if (!options.predictions_path.empty()) {
                    std::cout << "  Predictions:            " << options.predictions_path << "\n";
                }
                if (config.placement_policy == PlacementPolicy::RULE_BASED) {
                    std::cout << "  Rule threshold:         " << options.rule_threshold_us
                              << " us (min history " << options.rule_min_history << ")\n";
                }
            } else {
                std::cout << "  Working Set:            " << workload.lba_range_max << " blocks ("
                          << std::fixed << std::setprecision(2) << (actual_utilization * 100.0)
                          << "% of raw capacity)\n";
                std::cout << "NOTE: synthetic data is for development and tests only; "
                             "conference results use real traces.\n";
            }
            std::cout << "\nExecuting simulation...\n";
        }

        LfsSimulator sim(config);
        ReplayCounters counters;

        std::unique_ptr<TraceReader> trace_reader;
        std::unique_ptr<PredictionReader> prediction_reader;
        std::unique_ptr<RuleBasedClassifier> rule_classifier;

        if (config.placement_policy == PlacementPolicy::RULE_BASED) {
            rule_classifier = std::make_unique<RuleBasedClassifier>(options.rule_threshold_us,
                                                                   options.rule_min_history);
        }
        if (!options.predictions_path.empty()) {
            prediction_reader = std::make_unique<PredictionReader>(options.predictions_path);
        }

        auto temperature_for = [&](uint64_t timestamp, LbaType lba) -> Temperature {
            switch (config.placement_policy) {
                case PlacementPolicy::RULE_BASED: {
                    const Temperature t = rule_classifier->classify(lba, timestamp);
                    rule_classifier->observe(lba, timestamp);
                    return t;
                }
                case PlacementPolicy::LSTM_SMARTGC:
                    return prediction_reader->lookup(timestamp, lba);
                case PlacementPolicy::MIXED:
                default:
                    return Temperature::UNKNOWN;
            }
        };

        auto account = [&counters](Temperature t) {
            if (t == Temperature::HOT) counters.hot_placements++;
            else if (t == Temperature::COLD) counters.cold_placements++;
            else counters.unknown_placements++;
        };

        if (replay_trace) {
            trace_reader = std::make_unique<TraceReader>(options.trace_path);
            TraceRecord record;
            while (trace_reader->next(record)) {
                if (record.operation == 'R') {
                    counters.read_requests++;
                    continue;
                }
                if (record.operation == 'D') {
                    counters.discard_requests++;
                    continue;
                }
                const Temperature temperature = temperature_for(record.timestamp, record.lba);
                account(temperature);
                sim.write(record.lba, config.block_size_bytes, temperature);
                counters.write_requests++;

                if (options.max_writes && counters.write_requests >= options.max_writes) {
                    break;
                }
                if (!options.quiet && counters.write_requests % 250000 == 0) {
                    std::cout << "\r  " << counters.write_requests << " writes, WAF "
                              << std::fixed << std::setprecision(4) << sim.metrics().waf()
                              << "   " << std::flush;
                }
            }
            if (prediction_reader) {
                prediction_reader->finalize();
            }
            if (trace_label.empty()) {
                trace_label = trace_reader->trace_id().empty() ? "trace" : trace_reader->trace_id();
            }
        } else {
            for (const WorkloadRequest& request : synthetic_requests) {
                const Temperature temperature = temperature_for(request.timestamp, request.lba);
                account(temperature);
                sim.write(request.lba, config.block_size_bytes, temperature);
                counters.write_requests++;
            }
        }

        if (!options.quiet) {
            std::cout << "\n\nSimulation completed successfully.\n";
            if (!options.quiet && !replay_trace) {
                sim.print_status_summary();
            }
        }

        const StorageMetrics& metrics = sim.metrics();
        if (replay_trace) {
            actual_utilization = static_cast<double>(sim.live_blocks())
                                 / static_cast<double>(config.total_capacity_blocks());
        }

        if (!options.quiet) {
            std::cout << metrics.to_summary_string() << "\n";
            std::cout << " Final live utilization:  " << std::fixed << std::setprecision(2)
                      << (sim.utilization() * 100.0) << "% of raw capacity ("
                      << sim.live_blocks() << " live blocks)\n";
            if (trace_reader) {
                std::cout << " Trace rows read:         " << trace_reader->rows_read()
                          << "  (reads " << counters.read_requests
                          << ", discards " << counters.discard_requests
                          << ", malformed " << trace_reader->malformed_rows()
                          << ", bad-size " << trace_reader->non_unit_size_rows() << ")\n";
            }
            if (prediction_reader) {
                std::cout << " Predictions matched:     " << prediction_reader->rows_matched()
                          << " of " << prediction_reader->rows_read() << " rows"
                          << "  (lookup misses " << prediction_reader->lookup_misses()
                          << ", unmatched " << prediction_reader->unmatched_predictions()
                          << ", malformed " << prediction_reader->malformed_rows() << ")\n";
            }
            if (rule_classifier) {
                std::cout << " Rule classifications:    HOT " << rule_classifier->classified_hot()
                          << ", COLD " << rule_classifier->classified_cold()
                          << ", UNKNOWN " << rule_classifier->classified_unknown() << "\n";
            }
            std::cout << " Placement split:         HOT " << counters.hot_placements
                      << ", COLD " << counters.cold_placements
                      << ", UNKNOWN " << counters.unknown_placements << "\n\n";
        }

        if (!options.export_metrics.empty()) {
            bool file_exists = false;
            {
                std::ifstream test(options.export_metrics);
                file_exists = test.good();
            }
            std::ofstream out(options.export_metrics, std::ios::app);
            if (!out.is_open()) {
                std::cerr << "Error: cannot open metrics file for writing: "
                          << options.export_metrics << "\n";
                return 4;
            }
            if (!file_exists) {
                out << StorageMetrics::csv_header() << "\n";
            }

            RunDescriptor run;
            run.dataset = options.dataset;
            run.trace = trace_label;
            run.policy = placement_policy_to_string(config.placement_policy);
            run.model_type = options.model_type;
            run.total_segments = config.total_segments;
            run.blocks_per_segment = config.blocks_per_segment;
            run.block_size_bytes = config.block_size_bytes;
            run.gc_reserved_segments = config.gc_reserved_segments;
            run.gc_separate_stream = config.gc_separate_stream;
            run.target_utilization = actual_utilization;
            run.seed = config.random_seed;
            out << run.to_csv_row(metrics) << "\n";
        }

        if (!options.report_json.empty()) {
            std::ofstream out(options.report_json);
            if (!out.is_open()) {
                std::cerr << "Error: cannot open report file: " << options.report_json << "\n";
                return 4;
            }
            out << std::boolalpha;
            out << "{\n"
                << "  \"dataset\": \"" << json_escape(options.dataset) << "\",\n"
                << "  \"trace\": \"" << json_escape(trace_label) << "\",\n"
                << "  \"policy\": \"" << placement_policy_to_string(config.placement_policy) << "\",\n"
                << "  \"model_type\": \"" << json_escape(options.model_type) << "\",\n"
                << "  \"total_segments\": " << config.total_segments << ",\n"
                << "  \"blocks_per_segment\": " << config.blocks_per_segment << ",\n"
                << "  \"block_size_bytes\": " << config.block_size_bytes << ",\n"
                << "  \"gc_reserved_segments\": " << config.gc_reserved_segments << ",\n"
                << "  \"gc_separate_stream\": " << config.gc_separate_stream << ",\n"
                << "  \"seed\": " << config.random_seed << ",\n"
                << "  \"write_requests\": " << counters.write_requests << ",\n"
                << "  \"read_requests\": " << counters.read_requests << ",\n"
                << "  \"discard_requests\": " << counters.discard_requests << ",\n"
                << "  \"hot_placements\": " << counters.hot_placements << ",\n"
                << "  \"cold_placements\": " << counters.cold_placements << ",\n"
                << "  \"unknown_placements\": " << counters.unknown_placements << ",\n"
                << "  \"logical_bytes_written\": " << metrics.logical_bytes_written << ",\n"
                << "  \"physical_bytes_written\": " << metrics.physical_bytes_written << ",\n"
                << "  \"gc_bytes_copied\": " << metrics.gc_bytes_copied << ",\n"
                << "  \"valid_blocks_migrated\": " << metrics.valid_blocks_migrated << ",\n"
                << "  \"gc_count\": " << metrics.gc_count << ",\n"
                << "  \"live_blocks\": " << sim.live_blocks() << ",\n"
                << "  \"final_utilization\": " << std::setprecision(6) << sim.utilization() << ",\n"
                << "  \"waf\": " << std::setprecision(6) << metrics.waf();
            if (trace_reader) {
                out << ",\n  \"trace_rows_read\": " << trace_reader->rows_read()
                    << ",\n  \"trace_malformed_rows\": " << trace_reader->malformed_rows()
                    << ",\n  \"trace_non_unit_size_rows\": " << trace_reader->non_unit_size_rows()
                    << ",\n  \"trace_unknown_operation_rows\": " << trace_reader->unknown_operation_rows();
            }
            if (prediction_reader) {
                out << ",\n  \"prediction_rows_read\": " << prediction_reader->rows_read()
                    << ",\n  \"prediction_rows_matched\": " << prediction_reader->rows_matched()
                    << ",\n  \"prediction_lookup_misses\": " << prediction_reader->lookup_misses()
                    << ",\n  \"prediction_unmatched\": " << prediction_reader->unmatched_predictions()
                    << ",\n  \"prediction_malformed_rows\": " << prediction_reader->malformed_rows()
                    << ",\n  \"prediction_duplicate_keys\": " << prediction_reader->duplicate_key_rows()
                    << ",\n  \"prediction_out_of_order_input\": " << prediction_reader->out_of_order_input()
                    << ",\n  \"prediction_model_version\": \""
                    << json_escape(prediction_reader->model_version()) << "\"";
            }
            if (rule_classifier) {
                out << ",\n  \"rule_threshold_us\": " << options.rule_threshold_us
                    << ",\n  \"rule_min_history\": " << options.rule_min_history
                    << ",\n  \"rule_classified_hot\": " << rule_classifier->classified_hot()
                    << ",\n  \"rule_classified_cold\": " << rule_classifier->classified_cold()
                    << ",\n  \"rule_classified_unknown\": " << rule_classifier->classified_unknown();
            }
            out << "\n}\n";
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
