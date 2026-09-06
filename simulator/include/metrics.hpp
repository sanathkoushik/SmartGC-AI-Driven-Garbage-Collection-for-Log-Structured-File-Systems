#ifndef SMARTGC_METRICS_HPP
#define SMARTGC_METRICS_HPP

#include <cstdint>
#include <string>
#include <sstream>
#include <iomanip>

namespace smartgc {

struct StorageMetrics {
    uint64_t logical_bytes_written = 0;
    uint64_t physical_bytes_written = 0;
    uint64_t gc_bytes_copied = 0;
    uint64_t valid_blocks_migrated = 0;
    uint64_t gc_count = 0;
    uint64_t total_write_requests = 0;

    double waf() const {
        if (logical_bytes_written == 0) {
            return 1.0;
        }
        return static_cast<double>(physical_bytes_written) / static_cast<double>(logical_bytes_written);
    }

    void reset() {
        logical_bytes_written = 0;
        physical_bytes_written = 0;
        gc_bytes_copied = 0;
        valid_blocks_migrated = 0;
        gc_count = 0;
        total_write_requests = 0;
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
            << " Write Amplification (WAF): " << std::fixed << std::setprecision(4) << waf() << "\n"
            << "--------------------------------------------------";
        return oss.str();
    }

    static std::string csv_header() {
        return "workload_name,placement_policy,total_segments,blocks_per_segment,"
               "logical_bytes_written,physical_bytes_written,gc_bytes_copied,"
               "valid_blocks_migrated,gc_count,waf";
    }

    std::string to_csv_row(const std::string& workload_name,
                           const std::string& placement_policy_str,
                           size_t total_segments,
                           size_t blocks_per_segment) const {
        std::ostringstream oss;
        oss << workload_name << ","
            << placement_policy_str << ","
            << total_segments << ","
            << blocks_per_segment << ","
            << logical_bytes_written << ","
            << physical_bytes_written << ","
            << gc_bytes_copied << ","
            << valid_blocks_migrated << ","
            << gc_count << ","
            << std::fixed << std::setprecision(6) << waf();
        return oss.str();
    }
};

} // namespace smartgc

#endif // SMARTGC_METRICS_HPP
