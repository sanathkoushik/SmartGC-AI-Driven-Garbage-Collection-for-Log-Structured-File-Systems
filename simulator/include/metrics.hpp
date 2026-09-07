#ifndef SMARTGC_METRICS_HPP
#define SMARTGC_METRICS_HPP

#include "types.hpp"
#include <cstdint>
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>

namespace smartgc {

// Extra run context that is not part of the physical-write accounting but is
// persisted alongside every metrics row so runs are reproducible/comparable.
struct RunContext {
    std::string workload_name = "unknown";
    std::string placement_policy = "MIXED";
    std::string run_id = "";              // hash of the effective config (set by orchestrator)
    size_t total_segments = 0;
    size_t blocks_per_segment = 0;
    size_t migration_stream_count = 1;
    bool learned_trigger_enabled = false;
    double avg_gc_decision_latency_us = 0.0;
};

struct StorageMetrics {
    uint64_t logical_bytes_written = 0;
    uint64_t physical_bytes_written = 0;
    uint64_t gc_bytes_copied = 0;
    uint64_t valid_blocks_migrated = 0;
    uint64_t gc_count = 0;
    uint64_t total_write_requests = 0;

    // Phase 4b: confidence gate accounting.
    uint64_t predicted_writes = 0;          // writes that consulted a prediction row
    uint64_t confidence_fallbacks = 0;      // of those, how many fell back to RULE_BASED

    // Phase 7: per-GC-invocation migrated-block counts, for tail approximation.
    std::vector<uint64_t> per_gc_migrated;

    double waf() const {
        if (logical_bytes_written == 0) {
            return 1.0;
        }
        return static_cast<double>(physical_bytes_written) / static_cast<double>(logical_bytes_written);
    }

    double confidence_fallback_rate() const {
        if (predicted_writes == 0) return 0.0;
        return static_cast<double>(confidence_fallbacks) / static_cast<double>(predicted_writes);
    }

    double gc_migrated_percentile(double p) const {
        if (per_gc_migrated.empty()) return 0.0;
        std::vector<uint64_t> v = per_gc_migrated;
        std::sort(v.begin(), v.end());
        if (p <= 0.0) return static_cast<double>(v.front());
        if (p >= 1.0) return static_cast<double>(v.back());
        const double rank = p * (static_cast<double>(v.size()) - 1.0);
        const size_t lo = static_cast<size_t>(std::floor(rank));
        const size_t hi = static_cast<size_t>(std::ceil(rank));
        const double frac = rank - static_cast<double>(lo);
        return static_cast<double>(v[lo]) + frac * (static_cast<double>(v[hi]) - static_cast<double>(v[lo]));
    }

    void reset() {
        logical_bytes_written = 0;
        physical_bytes_written = 0;
        gc_bytes_copied = 0;
        valid_blocks_migrated = 0;
        gc_count = 0;
        total_write_requests = 0;
        predicted_writes = 0;
        confidence_fallbacks = 0;
        per_gc_migrated.clear();
    }

    std::string to_summary_string() const {
        std::ostringstream oss;
        oss << "--------------------------------------------------\n"
            << " SmartGC Simulator Metrics Summary\n"
            << "--------------------------------------------------\n"
            << " Total Write Requests:    " << total_write_requests << "\n"
            << " Logical Bytes Written:   " << logical_bytes_written << " bytes ("
            << (logical_bytes_written / 1024.0 / 1024.0) << " MB)\n"
            << " Physical Bytes Written:  " << physical_bytes_written << " bytes ("
            << (physical_bytes_written / 1024.0 / 1024.0) << " MB)\n"
            << " GC Migrated Bytes:       " << gc_bytes_copied << " bytes ("
            << (gc_bytes_copied / 1024.0 / 1024.0) << " MB)\n"
            << " Valid Blocks Migrated:   " << valid_blocks_migrated << "\n"
            << " GC Invocation Count:     " << gc_count << "\n"
            << " GC Migrated / invocation P50/P95/P99: "
            << gc_migrated_percentile(0.50) << " / "
            << gc_migrated_percentile(0.95) << " / "
            << gc_migrated_percentile(0.99) << "\n"
            << " Confidence Fallback Rate: " << std::fixed << std::setprecision(4)
            << confidence_fallback_rate() << " (" << confidence_fallbacks << "/" << predicted_writes << ")\n"
            << " Write Amplification (WAF): " << std::fixed << std::setprecision(4) << waf() << "\n"
            << "--------------------------------------------------";
        return oss.str();
    }

    // Metrics contract v2 (docs/architecture.md 2.3).
    static std::string csv_header() {
        return "contract_version,run_id,workload_name,placement_policy,total_segments,blocks_per_segment,"
               "logical_bytes_written,physical_bytes_written,gc_bytes_copied,"
               "valid_blocks_migrated,gc_count,waf,"
               "migration_stream_count,learned_trigger_enabled,avg_gc_decision_latency_us,"
               "gc_migrated_p50,gc_migrated_p95,gc_migrated_p99,"
               "predicted_writes,confidence_fallbacks,confidence_fallback_rate";
    }

    std::string to_csv_row(const RunContext& ctx) const {
        std::ostringstream oss;
        oss << CONTRACT_VERSION << ","
            << ctx.run_id << ","
            << ctx.workload_name << ","
            << ctx.placement_policy << ","
            << ctx.total_segments << ","
            << ctx.blocks_per_segment << ","
            << logical_bytes_written << ","
            << physical_bytes_written << ","
            << gc_bytes_copied << ","
            << valid_blocks_migrated << ","
            << gc_count << ","
            << std::fixed << std::setprecision(6) << waf() << ","
            << ctx.migration_stream_count << ","
            << (ctx.learned_trigger_enabled ? 1 : 0) << ","
            << std::fixed << std::setprecision(3) << ctx.avg_gc_decision_latency_us << ","
            << std::setprecision(3)
            << gc_migrated_percentile(0.50) << ","
            << gc_migrated_percentile(0.95) << ","
            << gc_migrated_percentile(0.99) << ","
            << predicted_writes << ","
            << confidence_fallbacks << ","
            << std::setprecision(6) << confidence_fallback_rate();
        return oss.str();
    }
};

} // namespace smartgc

#endif // SMARTGC_METRICS_HPP
