#ifndef SMARTGC_TRACE_READER_HPP
#define SMARTGC_TRACE_READER_HPP

#include "types.hpp"
#include <cstdint>
#include <fstream>
#include <string>

namespace smartgc {

struct TraceRecord {
    uint64_t timestamp = 0;
    LbaType lba = INVALID_LBA;
    uint32_t size_blocks = 1;
    char operation = 'W';
};

// Streaming reader for the normalized trace contract (docs/architecture.md 2.1):
//
//     timestamp,lba,size,operation,trace_id
//
// Streaming rather than loading the file: a normalized MSR volume can hold tens
// of millions of rows, and the simulator only ever needs one at a time.
//
// Malformed rows are counted, never silently skipped without trace: the counts
// are reported at the end of every run so a parsing problem cannot masquerade
// as a workload property.
class TraceReader {
public:
    explicit TraceReader(const std::string& path);

    // Reads the next record. Returns false at end of file.
    bool next(TraceRecord& out);

    const std::string& path() const { return path_; }
    const std::string& trace_id() const { return trace_id_; }
    uint64_t rows_read() const { return rows_read_; }
    uint64_t malformed_rows() const { return malformed_rows_; }
    uint64_t unknown_operation_rows() const { return unknown_operation_rows_; }
    uint64_t negative_lba_rows() const { return negative_lba_rows_; }
    uint64_t non_unit_size_rows() const { return non_unit_size_rows_; }

private:
    bool parse_line(const std::string& line, TraceRecord& out);

    std::string path_;
    std::ifstream stream_;
    std::string trace_id_;
    uint64_t rows_read_ = 0;
    uint64_t malformed_rows_ = 0;
    uint64_t unknown_operation_rows_ = 0;
    uint64_t negative_lba_rows_ = 0;
    uint64_t non_unit_size_rows_ = 0;
};

} // namespace smartgc

#endif // SMARTGC_TRACE_READER_HPP
