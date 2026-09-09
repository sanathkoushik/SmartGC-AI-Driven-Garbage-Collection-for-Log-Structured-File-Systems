#include "config.hpp"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace smartgc {
namespace {

std::string trim(const std::string& s) {
    const std::string ws = " \t\r\n";
    const size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) {
        return "";
    }
    const size_t e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

// Strips a trailing YAML comment. A '#' only starts a comment when it is outside
// quotes and preceded by whitespace or start-of-line, so a value such as a#b
// survives intact.
std::string strip_comment(const std::string& line) {
    bool in_single = false;
    bool in_double = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '\'' && !in_double) {
            in_single = !in_single;
        } else if (c == '"' && !in_single) {
            in_double = !in_double;
        } else if (c == '#' && !in_single && !in_double) {
            if (i == 0 || line[i - 1] == ' ' || line[i - 1] == '\t') {
                return line.substr(0, i);
            }
        }
    }
    return line;
}

std::string unquote(const std::string& s) {
    if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') ||
                          (s.front() == '\'' && s.back() == '\''))) {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

using Section = std::map<std::string, std::string>;

// Parses the strict subset of YAML that the SmartGC config.yaml uses: top-level
// scalars and one level of section blocks containing scalars. Anything else
// (sequences, deeper nesting, tab indentation) is rejected with a message that
// names the offending line, so a malformed config never degrades into defaults.
std::map<std::string, Section> parse_flat_yaml(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open config file: " + path);
    }

    std::map<std::string, Section> doc;
    doc[""] = Section{};                 // "" holds the top-level scalars
    std::string current_section;
    std::string raw;
    size_t line_no = 0;

    while (std::getline(in, raw)) {
        ++line_no;
        const std::string line = strip_comment(raw);
        if (trim(line).empty()) {
            continue;
        }
        if (line.find('\t') != std::string::npos) {
            throw std::runtime_error(path + ":" + std::to_string(line_no) +
                                     ": tabs are not valid YAML indentation");
        }

        const size_t indent = line.find_first_not_of(' ');
        const std::string body = trim(line);

        if (body.front() == '-') {
            throw std::runtime_error(path + ":" + std::to_string(line_no) +
                                     ": YAML sequences are not supported by the SmartGC config loader");
        }

        const size_t colon = body.find(':');
        if (colon == std::string::npos) {
            throw std::runtime_error(path + ":" + std::to_string(line_no) +
                                     ": expected 'key: value', got '" + body + "'");
        }

        const std::string key = trim(body.substr(0, colon));
        const std::string value = unquote(trim(body.substr(colon + 1)));

        if (key.empty()) {
            throw std::runtime_error(path + ":" + std::to_string(line_no) + ": empty key");
        }

        if (indent == 0) {
            if (value.empty()) {
                current_section = key;
                doc[current_section];        // create the section
            } else {
                current_section.clear();
                doc[""][key] = value;
            }
            continue;
        }

        if (current_section.empty()) {
            throw std::runtime_error(path + ":" + std::to_string(line_no) +
                                     ": indented key '" + key + "' has no enclosing section");
        }
        if (value.empty()) {
            throw std::runtime_error(path + ":" + std::to_string(line_no) +
                                     ": nested mappings deeper than one level are not supported ('" + key + "')");
        }
        doc[current_section][key] = value;
    }

    return doc;
}

size_t to_size(const std::string& key, const std::string& value) {
    try {
        const long long v = std::stoll(value);
        if (v < 0) {
            throw std::out_of_range("negative");
        }
        return static_cast<size_t>(v);
    } catch (const std::exception&) {
        throw std::runtime_error("config key '" + key + "': expected a non-negative integer, got '" + value + "'");
    }
}

long long to_ll(const std::string& key, const std::string& value) {
    try {
        return std::stoll(value);
    } catch (const std::exception&) {
        throw std::runtime_error("config key '" + key + "': expected an integer, got '" + value + "'");
    }
}

double to_double(const std::string& key, const std::string& value) {
    try {
        return std::stod(value);
    } catch (const std::exception&) {
        throw std::runtime_error("config key '" + key + "': expected a number, got '" + value + "'");
    }
}

bool to_bool(const std::string& key, const std::string& value) {
    if (value == "true" || value == "True" || value == "yes" || value == "1") return true;
    if (value == "false" || value == "False" || value == "no" || value == "0") return false;
    throw std::runtime_error("config key '" + key + "': expected a boolean, got '" + value + "'");
}

void reject_unknown_keys(const Section& section,
                         const std::set<std::string>& known,
                         const std::string& section_name) {
    for (const auto& kv : section) {
        if (known.find(kv.first) == known.end()) {
            throw std::runtime_error("Unknown key '" + section_name + "." + kv.first +
                                     "' in SmartGC config; refusing to fall back to a default");
        }
    }
}

const std::string* lookup(const Section& s, const char* key) {
    const auto f = s.find(key);
    return f == s.end() ? nullptr : &f->second;
}

} // namespace

void SimulatorConfig::validate() const {
    if (block_size_bytes == 0) {
        throw std::invalid_argument("simulator.block_size_bytes must be > 0");
    }
    if (blocks_per_segment == 0) {
        throw std::invalid_argument("simulator.blocks_per_segment must be > 0");
    }
    if (total_segments == 0) {
        throw std::invalid_argument("simulator.total_segments must be > 0");
    }
    if (gc_reserved_segments < 1) {
        throw std::invalid_argument(
            "simulator.gc_reserved_segments must be >= 1: without a reserve a cleaning pass "
            "can exhaust the free pool while migrating a victim's valid blocks");
    }
    const size_t needed = unavailable_segments() + 1;
    if (total_segments < needed) {
        std::ostringstream oss;
        oss << "simulator.total_segments (" << total_segments << ") is too small: the GC reserve ("
            << gc_reserved_segments << ") plus " << user_stream_count()
            << " user append point(s) plus " << (gc_separate_stream ? 1 : 0)
            << " GC append point(s) already needs " << unavailable_segments()
            << " segments, leaving none to hold reclaimable data. Use at least "
            << needed << " segments.";
        throw std::invalid_argument(oss.str());
    }
    if (gc_free_segments_threshold >= total_segments) {
        throw std::invalid_argument("simulator.gc_free_segments_threshold must be < simulator.total_segments");
    }
}

std::string SimulatorConfig::to_summary_string() const {
    std::ostringstream oss;
    oss << "  Total Segments:         " << total_segments << "\n"
        << "  Blocks per Segment:     " << blocks_per_segment << "\n"
        << "  Block Size:             " << block_size_bytes << " bytes\n"
        << "  Raw Capacity:           " << std::fixed << std::setprecision(2)
        << (total_capacity_bytes() / 1024.0 / 1024.0) << " MB (" << total_capacity_blocks() << " blocks)\n"
        << "  GC Trigger Threshold:   " << gc_free_segments_threshold << " free segments\n"
        << "  GC Reserved Segments:   " << gc_reserved_segments << "\n"
        << "  GC Separate Stream:     " << (gc_separate_stream ? "yes" : "no") << "\n"
        << "  Placement Policy:       " << placement_policy_to_string(placement_policy) << "\n"
        << "  User Append Points:     " << user_stream_count() << "\n"
        << "  GC Policy:              " << gc_policy_to_string(gc_policy) << "\n"
        << "  Max Live Blocks:        " << max_live_blocks() << " ("
        << std::fixed << std::setprecision(2) << (max_utilization() * 100.0) << "% of raw capacity)\n"
        << "  Random Seed:            " << random_seed << "\n";
    return oss.str();
}

void WorkloadConfig::validate() const {
    if (total_requests == 0) {
        throw std::invalid_argument("synthetic_workload.total_requests must be > 0");
    }
    if (lba_range_max <= lba_range_min) {
        throw std::invalid_argument("synthetic_workload.lba_range_max must be > synthetic_workload.lba_range_min");
    }
    if (hot_ratio <= 0.0 || hot_ratio >= 1.0) {
        throw std::invalid_argument("synthetic_workload.hot_ratio must be in (0, 1)");
    }
    if (hot_traffic_ratio < 0.0 || hot_traffic_ratio > 1.0) {
        throw std::invalid_argument("synthetic_workload.hot_traffic_ratio must be in [0, 1]");
    }
}

SimulatorConfig load_simulator_config(const std::string& yaml_path) {
    const auto doc = parse_flat_yaml(yaml_path);
    SimulatorConfig cfg;

    const auto top = doc.find("");
    if (top != doc.end()) {
        if (const std::string* seed = lookup(top->second, "random_seed")) {
            cfg.random_seed = static_cast<uint32_t>(to_size("random_seed", *seed));
        }
    }

    const auto it = doc.find("simulator");
    if (it == doc.end()) {
        throw std::runtime_error("Config file '" + yaml_path + "' has no 'simulator:' section");
    }
    const Section& s = it->second;

    reject_unknown_keys(s, {
        "block_size_bytes", "blocks_per_segment", "total_segments",
        "gc_free_segments_threshold", "gc_reserved_segments", "gc_separate_stream",
        "gc_policy", "placement_policy"
    }, "simulator");

    if (const std::string* v = lookup(s, "block_size_bytes"))            cfg.block_size_bytes = to_size("simulator.block_size_bytes", *v);
    if (const std::string* v = lookup(s, "blocks_per_segment"))          cfg.blocks_per_segment = to_size("simulator.blocks_per_segment", *v);
    if (const std::string* v = lookup(s, "total_segments"))              cfg.total_segments = to_size("simulator.total_segments", *v);
    if (const std::string* v = lookup(s, "gc_free_segments_threshold"))  cfg.gc_free_segments_threshold = to_size("simulator.gc_free_segments_threshold", *v);
    if (const std::string* v = lookup(s, "gc_reserved_segments"))        cfg.gc_reserved_segments = to_size("simulator.gc_reserved_segments", *v);
    if (const std::string* v = lookup(s, "gc_separate_stream"))          cfg.gc_separate_stream = to_bool("simulator.gc_separate_stream", *v);

    if (const std::string* v = lookup(s, "gc_policy")) {
        if (*v != "GREEDY") {
            throw std::runtime_error("simulator.gc_policy: only 'GREEDY' is implemented, got '" + *v + "'");
        }
        cfg.gc_policy = GcPolicy::GREEDY;
    }
    if (const std::string* v = lookup(s, "placement_policy")) {
        if (!parse_placement_policy(*v, cfg.placement_policy)) {
            throw std::runtime_error("simulator.placement_policy: expected MIXED, RULE_BASED or LSTM_SMARTGC, got '" + *v + "'");
        }
    }

    return cfg;
}

WorkloadConfig load_workload_config(const std::string& yaml_path) {
    const auto doc = parse_flat_yaml(yaml_path);
    WorkloadConfig cfg;

    const auto it = doc.find("synthetic_workload");
    if (it == doc.end()) {
        return cfg;
    }
    const Section& s = it->second;

    reject_unknown_keys(s, {
        "total_requests", "lba_range_min", "lba_range_max", "hot_ratio", "hot_traffic_ratio"
    }, "synthetic_workload");

    if (const std::string* v = lookup(s, "total_requests"))     cfg.total_requests = to_size("synthetic_workload.total_requests", *v);
    if (const std::string* v = lookup(s, "lba_range_min"))      cfg.lba_range_min = to_ll("synthetic_workload.lba_range_min", *v);
    if (const std::string* v = lookup(s, "lba_range_max"))      cfg.lba_range_max = to_ll("synthetic_workload.lba_range_max", *v);
    if (const std::string* v = lookup(s, "hot_ratio"))          cfg.hot_ratio = to_double("synthetic_workload.hot_ratio", *v);
    if (const std::string* v = lookup(s, "hot_traffic_ratio"))  cfg.hot_traffic_ratio = to_double("synthetic_workload.hot_traffic_ratio", *v);

    return cfg;
}

} // namespace smartgc
