#ifndef SMARTGC_RULE_BASED_CLASSIFIER_HPP
#define SMARTGC_RULE_BASED_CLASSIFIER_HPP

#include "types.hpp"
#include <cstdint>
#include <unordered_map>

namespace smartgc {

// The non-ML control: a running mean of an LBA's observed rewrite intervals,
// compared against a fixed threshold.
//
//     HOT   if mean(observed rewrite intervals of this LBA) <= threshold
//     COLD  otherwise
//
// Deliberately simple. It holds two numbers per LBA and does one comparison,
// with no learned parameters, no sequence model and no lookahead - if it were
// any more elaborate it would be a hidden ML model and the comparison would
// stop meaning anything.
//
// Two design points that keep the comparison controlled:
//
//  * The threshold is supplied by the caller and comes from the *training*
//    portion of the workload, exactly like the LSTM's threshold. Deriving it
//    from the data being replayed would be fitting on the test set.
//
//  * `min_history` mirrors the model's sequence length, so the heuristic stays
//    silent (UNKNOWN) for the same early writes the LSTM cannot score. Without
//    it the two policies would differ in prediction *coverage* as well as
//    prediction *quality*, and the WAF difference could not be attributed.
class RuleBasedClassifier {
public:
    RuleBasedClassifier(double hot_threshold, size_t min_history)
        : hot_threshold_(hot_threshold), min_history_(min_history) {}

    // Temperature for a write to `lba` at `timestamp`, from history only.
    // Returns UNKNOWN until the LBA has at least `min_history` intervals.
    Temperature classify(LbaType lba, uint64_t timestamp) const;

    // Records the write so it can inform later classifications. Must be called
    // once per write, after classify().
    void observe(LbaType lba, uint64_t timestamp);

    double hot_threshold() const { return hot_threshold_; }
    size_t min_history() const { return min_history_; }
    size_t tracked_lbas() const { return history_.size(); }
    uint64_t classified_hot() const { return classified_hot_; }
    uint64_t classified_cold() const { return classified_cold_; }
    uint64_t classified_unknown() const { return classified_unknown_; }

private:
    struct History {
        uint64_t last_write_timestamp = 0;
        uint64_t interval_count = 0;
        double interval_sum = 0.0;
    };

    double hot_threshold_;
    size_t min_history_;
    std::unordered_map<LbaType, History> history_;

    mutable uint64_t classified_hot_ = 0;
    mutable uint64_t classified_cold_ = 0;
    mutable uint64_t classified_unknown_ = 0;
};

} // namespace smartgc

#endif // SMARTGC_RULE_BASED_CLASSIFIER_HPP
