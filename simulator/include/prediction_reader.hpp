#ifndef SMARTGC_PREDICTION_READER_HPP
#define SMARTGC_PREDICTION_READER_HPP

#include "types.hpp"
#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>

namespace smartgc {

// Streaming merge-join against the prediction contract
// (docs/architecture.md 2.2):
//
//     timestamp,lba,predicted_rewrite_interval,predicted_class,model_version,trace_id
//
// Both the trace and the prediction file are in chronological order, so the
// reader advances a single cursor as the trace is replayed. That makes lookup
// O(1) amortised without holding tens of millions of predictions in memory.
//
// Not every write has a prediction: an LBA needs `sequence_length` rewrite
// intervals of history before the model can score it. A miss is a normal,
// counted outcome that yields Temperature::UNKNOWN - never a guess.
//
// Every anomaly is counted and reported: malformed rows, unknown class labels,
// predictions whose timestamp never appears in the trace, and duplicate
// (timestamp, lba) keys. Silence about these would let a broken prediction file
// look like a poorly performing model.
class PredictionReader {
public:
    explicit PredictionReader(const std::string& path);

    // Looks up the prediction for a write at (timestamp, lba).
    // `timestamp` must be non-decreasing across calls, matching trace order.
    // Returns Temperature::UNKNOWN when this write has no prediction.
    Temperature lookup(uint64_t timestamp, LbaType lba, double& predicted_interval);
    Temperature lookup(uint64_t timestamp, LbaType lba);

    // Predictions still buffered or unread once the replay finishes.
    void finalize();

    const std::string& path() const { return path_; }
    const std::string& model_version() const { return model_version_; }
    const std::string& trace_id() const { return trace_id_; }
    uint64_t rows_read() const { return rows_read_; }
    uint64_t rows_matched() const { return rows_matched_; }
    uint64_t lookups() const { return lookups_; }
    uint64_t lookup_misses() const { return lookup_misses_; }
    uint64_t malformed_rows() const { return malformed_rows_; }
    uint64_t unknown_class_rows() const { return unknown_class_rows_; }
    uint64_t duplicate_key_rows() const { return duplicate_key_rows_; }
    uint64_t unmatched_predictions() const { return unmatched_predictions_; }
    bool out_of_order_input() const { return out_of_order_input_; }

private:
    struct Entry {
        double interval = 0.0;
        Temperature temperature = Temperature::UNKNOWN;
    };

    bool read_next_row();
    void load_group_for(uint64_t timestamp);

    std::string path_;
    std::ifstream stream_;
    std::string model_version_;
    std::string trace_id_;

    // One-row lookahead into the prediction stream.
    bool has_pending_ = false;
    uint64_t pending_timestamp_ = 0;
    LbaType pending_lba_ = INVALID_LBA;
    Entry pending_entry_{};

    // All predictions sharing the current timestamp. A single request expands
    // into several LBAs at one timestamp, so a group, not a single row.
    std::unordered_map<LbaType, Entry> group_;
    uint64_t group_timestamp_ = 0;
    bool group_loaded_ = false;

    uint64_t last_query_timestamp_ = 0;
    bool has_queried_ = false;
    bool out_of_order_input_ = false;

    uint64_t rows_read_ = 0;
    uint64_t rows_matched_ = 0;
    uint64_t lookups_ = 0;
    uint64_t lookup_misses_ = 0;
    uint64_t malformed_rows_ = 0;
    uint64_t unknown_class_rows_ = 0;
    uint64_t duplicate_key_rows_ = 0;
    uint64_t unmatched_predictions_ = 0;
};

} // namespace smartgc

#endif // SMARTGC_PREDICTION_READER_HPP
