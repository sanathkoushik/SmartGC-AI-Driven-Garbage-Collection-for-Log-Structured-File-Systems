#include "prediction_reader.hpp"

#include <array>
#include <cstdlib>
#include <stdexcept>

namespace smartgc {
namespace {

size_t split_csv(const std::string& line, std::array<std::string, 6>& fields, size_t max_fields) {
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

PredictionReader::PredictionReader(const std::string& path)
    : path_(path), stream_(path) {
    if (!stream_.is_open()) {
        throw std::runtime_error("Cannot open predictions file: " + path);
    }

    std::string header;
    if (!std::getline(stream_, header)) {
        throw std::runtime_error("Predictions file is empty: " + path);
    }
    std::array<std::string, 6> fields;
    const size_t count = split_csv(header, fields, 6);
    if (count < 4 || trim(fields[0]) != "timestamp" || trim(fields[1]) != "lba"
        || trim(fields[2]) != "predicted_rewrite_interval"
        || trim(fields[3]) != "predicted_class") {
        throw std::runtime_error(
            "Unexpected header in " + path + "; expected 'timestamp,lba,"
            "predicted_rewrite_interval,predicted_class,model_version,trace_id' but found '"
            + header + "'");
    }

    has_pending_ = read_next_row();
}

bool PredictionReader::read_next_row() {
    std::string line;
    while (std::getline(stream_, line)) {
        if (line.empty() || line == "\r") {
            continue;
        }
        rows_read_++;

        std::array<std::string, 6> fields;
        const size_t count = split_csv(line, fields, 6);
        if (count < 4) {
            malformed_rows_++;
            continue;
        }

        char* end = nullptr;
        const unsigned long long timestamp = std::strtoull(fields[0].c_str(), &end, 10);
        if (end == fields[0].c_str() || *end != '\0') {
            malformed_rows_++;
            continue;
        }
        end = nullptr;
        const long long lba = std::strtoll(fields[1].c_str(), &end, 10);
        if (end == fields[1].c_str() || *end != '\0' || lba < 0) {
            malformed_rows_++;
            continue;
        }
        end = nullptr;
        const double interval = std::strtod(fields[2].c_str(), &end);
        if (end == fields[2].c_str() || *end != '\0') {
            malformed_rows_++;
            continue;
        }

        const std::string label = trim(fields[3]);
        Temperature temperature;
        if (label == "HOT" || label == "1") {
            temperature = Temperature::HOT;
        } else if (label == "COLD" || label == "0") {
            temperature = Temperature::COLD;
        } else {
            // An unreadable label must not become a silent COLD: skip the row
            // and let the write fall back to UNKNOWN, which is counted.
            unknown_class_rows_++;
            continue;
        }

        if (model_version_.empty() && count >= 5) {
            model_version_ = trim(fields[4]);
        }
        if (trace_id_.empty() && count >= 6) {
            trace_id_ = trim(fields[5]);
        }

        pending_timestamp_ = static_cast<uint64_t>(timestamp);
        pending_lba_ = static_cast<LbaType>(lba);
        pending_entry_ = Entry{interval, temperature};
        return true;
    }
    return false;
}

void PredictionReader::load_group_for(uint64_t timestamp) {
    group_.clear();
    group_timestamp_ = timestamp;
    group_loaded_ = true;

    // Predictions older than the current write can never be claimed: the trace
    // has moved past them.
    while (has_pending_ && pending_timestamp_ < timestamp) {
        unmatched_predictions_++;
        has_pending_ = read_next_row();
    }
    while (has_pending_ && pending_timestamp_ == timestamp) {
        const auto inserted = group_.emplace(pending_lba_, pending_entry_);
        if (!inserted.second) {
            // Two predictions for the same write: the join would be ambiguous,
            // so keep the first and count the collision.
            duplicate_key_rows_++;
        }
        has_pending_ = read_next_row();
    }
}

Temperature PredictionReader::lookup(uint64_t timestamp, LbaType lba, double& predicted_interval) {
    lookups_++;
    if (has_queried_ && timestamp < last_query_timestamp_) {
        // The merge-join assumes a non-decreasing trace; record the violation
        // rather than returning quietly wrong answers.
        out_of_order_input_ = true;
    }
    last_query_timestamp_ = timestamp;
    has_queried_ = true;

    if (!group_loaded_ || group_timestamp_ != timestamp) {
        load_group_for(timestamp);
    }

    const auto found = group_.find(lba);
    if (found == group_.end()) {
        lookup_misses_++;
        return Temperature::UNKNOWN;
    }
    predicted_interval = found->second.interval;
    const Temperature temperature = found->second.temperature;
    // Erased so a duplicate write to the same LBA at the same instant does not
    // reuse a prediction that was meant for one event only.
    group_.erase(found);
    rows_matched_++;
    return temperature;
}

Temperature PredictionReader::lookup(uint64_t timestamp, LbaType lba) {
    double ignored = 0.0;
    return lookup(timestamp, lba, ignored);
}

void PredictionReader::finalize() {
    unmatched_predictions_ += static_cast<uint64_t>(group_.size());
    group_.clear();
    while (has_pending_) {
        unmatched_predictions_++;
        has_pending_ = read_next_row();
    }
}

} // namespace smartgc
