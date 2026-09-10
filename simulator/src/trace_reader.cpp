#include "trace_reader.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

namespace smartgc {
namespace {

// Splits at most `max_fields` comma-separated fields without allocating a
// vector per row; the trace player runs this tens of millions of times.
size_t split_csv(const std::string& line, std::array<std::string, 5>& fields, size_t max_fields) {
    size_t count = 0;
    size_t start = 0;
    while (count < max_fields) {
        const size_t comma = line.find(',', start);
        if (comma == std::string::npos) {
            fields[count++] = line.substr(start);
            break;
        }
        fields[count++] = line.substr(start, comma - start);
        start = comma + 1;
    }
    return count;
}

std::string trim(const std::string& value) {
    const size_t begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const size_t end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

} // namespace

TraceReader::TraceReader(const std::string& path)
    : path_(path), stream_(path) {
    if (!stream_.is_open()) {
        throw std::runtime_error("Cannot open normalized trace: " + path);
    }

    std::string header;
    if (!std::getline(stream_, header)) {
        throw std::runtime_error("Normalized trace is empty: " + path);
    }
    std::array<std::string, 5> fields;
    const size_t count = split_csv(header, fields, 5);
    if (count < 4 || trim(fields[0]) != "timestamp" || trim(fields[1]) != "lba"
        || trim(fields[2]) != "size" || trim(fields[3]) != "operation") {
        throw std::runtime_error(
            "Unexpected header in " + path + "; expected "
            "'timestamp,lba,size,operation,trace_id' but found '" + header + "'");
    }
}

bool TraceReader::parse_line(const std::string& line, TraceRecord& out) {
    std::array<std::string, 5> fields;
    const size_t count = split_csv(line, fields, 5);
    if (count < 4) {
        malformed_rows_++;
        return false;
    }

    char* end = nullptr;
    const unsigned long long timestamp = std::strtoull(fields[0].c_str(), &end, 10);
    if (end == fields[0].c_str() || *end != '\0') {
        malformed_rows_++;
        return false;
    }

    end = nullptr;
    const long long lba = std::strtoll(fields[1].c_str(), &end, 10);
    if (end == fields[1].c_str() || *end != '\0') {
        malformed_rows_++;
        return false;
    }
    if (lba < 0) {
        negative_lba_rows_++;
        return false;
    }

    end = nullptr;
    const long long size = std::strtoll(fields[2].c_str(), &end, 10);
    if (end == fields[2].c_str() || *end != '\0' || size <= 0) {
        malformed_rows_++;
        return false;
    }
    if (size != 1) {
        // Preprocessing expands multi-block requests, so a size other than one
        // block means the file did not come from this pipeline.
        non_unit_size_rows_++;
        return false;
    }

    const std::string operation = trim(fields[3]);
    if (operation.size() != 1 || (operation[0] != 'W' && operation[0] != 'R' && operation[0] != 'D')) {
        unknown_operation_rows_++;
        return false;
    }

    if (trace_id_.empty() && count >= 5) {
        trace_id_ = trim(fields[4]);
    }

    out.timestamp = static_cast<uint64_t>(timestamp);
    out.lba = static_cast<LbaType>(lba);
    out.size_blocks = 1;
    out.operation = operation[0];
    return true;
}

bool TraceReader::next(TraceRecord& out) {
    std::string line;
    while (std::getline(stream_, line)) {
        if (line.empty() || line == "\r") {
            continue;
        }
        rows_read_++;
        if (parse_line(line, out)) {
            return true;
        }
    }
    return false;
}

} // namespace smartgc
