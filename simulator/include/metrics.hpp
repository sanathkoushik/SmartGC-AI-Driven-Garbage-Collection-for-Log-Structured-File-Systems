#ifndef SMARTGC_METRICS_HPP
#define SMARTGC_METRICS_HPP

#include <cstdint>
#include <string>
#include <sstream>
#include <iomanip>

namespace smartgc {

struct StorageMetrics {
    // Bytes written on behalf of the workload. GC migrations never contribute here.
    uint64_t logical_bytes_written = 0;
    // Every byte that reaches the media: user writes plus GC migrations.
    uint64_t physical_bytes_written = 0;
    uint64_t gc_bytes_copied = 0;
    uint64_t valid_blocks_migrated = 0;
    uint64_t gc_count = 0;
    uint64_t total_write_requests = 0;

    // Placement breakdown. Under MIXED every user write is counted as
    // main_stream_writes; under the segregated policies the split reflects the
    // predicted temperature that drove placement.
    uint64_t main_stream_writes = 0;
    uint64_t cold_stream_writes = 0;
    // User writes for which no prediction was available and the policy fallback
    // was used. Tracked separately so prediction coverage is never guessed at.
    uint64_t unpredicted_writes = 0;

    double waf() const {
        if (logical_bytes_written == 0) {
            return 1.0;
        }
        return static_cast<double>(physical_bytes_written) / static_cast<double>(logical_bytes_written);
    }

    void reset() {
        *this = StorageMetrics{};
    }

    std::string to_summary_string() const {
        std::ostringstream oss;
        oss << "--------------------------------------------------\n"
            << " SmartGC Simulator Metrics Summary\n"
            << "--------------------------------------------------\n"
            << " Total Write Requests:    " << total_write_requests << "\n"
            << " Logical Bytes Written:   " << logical_bytes_written << " bytes ("
            << std::fixed << std::setprecision(4) << (logical_bytes_written / 1024.0 / 1024.0) << " MB)\n"
            << " Physical Bytes Written:  " << physical_bytes_written << " bytes ("
            << (physical_bytes_written / 1024.0 / 1024.0) << " MB)\n"
            << " GC Migrated Bytes:       " << gc_bytes_copied << " bytes ("
            << (gc_bytes_copied / 1024.0 / 1024.0) << " MB)\n"
            << " Valid Blocks Migrated:   " << valid_blocks_migrated << "\n"
            << " GC Invocation Count:     " << gc_count << "\n"
            << " Main-Stream Writes:      " << main_stream_writes << "\n"
            << " Cold-Stream Writes:      " << cold_stream_writes << "\n"
            << " Writes w/o Prediction:   " << unpredicted_writes << "\n"
            << " Write Amplification (WAF): " << std::fixed << std::setprecision(6) << waf() << "\n"
            << "--------------------------------------------------";
        return oss.str();
    }

    static std::string csv_header() {
        return "dataset,trace,policy,model_type,total_segments,blocks_per_segment,block_size_bytes,"
               "gc_reserved_segments,gc_separate_stream,target_utilization,seed,"
               "total_write_requests,logical_bytes_written,physical_bytes_written,gc_bytes_copied,"
               "valid_blocks_migrated,gc_count,main_stream_writes,cold_stream_writes,"
               "unpredicted_writes,waf";
    }
};

// Everything needed to identify one simulation run in results/metrics/*.csv.
// Kept separate from StorageMetrics so the counters stay pure measurements.
struct RunDescriptor {
    std::string dataset = "synthetic";
    std::string trace = "";
    std::string policy = "MIXED";
    std::string model_type = "none";
    size_t total_segments = 0;
    size_t blocks_per_segment = 0;
    size_t block_size_bytes = 0;
    size_t gc_reserved_segments = 0;
    bool gc_separate_stream = true;
    double target_utilization = 0.0;
    uint32_t seed = 0;

    std::string to_csv_row(const StorageMetrics& m) const {
        std::ostringstream oss;
        oss << dataset << ','
            << trace << ','
            << policy << ','
            << model_type << ','
            << total_segments << ','
            << blocks_per_segment << ','
            << block_size_bytes << ','
            << gc_reserved_segments << ','
            << (gc_separate_stream ? 1 : 0) << ','
            << std::fixed << std::setprecision(6) << target_utilization << ','
            << seed << ','
            << m.total_write_requests << ','
            << m.logical_bytes_written << ','
            << m.physical_bytes_written << ','
            << m.gc_bytes_copied << ','
            << m.valid_blocks_migrated << ','
            << m.gc_count << ','
            << m.main_stream_writes << ','
            << m.cold_stream_writes << ','
            << m.unpredicted_writes << ','
            << std::fixed << std::setprecision(6) << m.waf();
        return oss.str();
    }
};

} // namespace smartgc

#endif // SMARTGC_METRICS_HPP
