#include "rule_based_classifier.hpp"

namespace smartgc {

Temperature RuleBasedClassifier::classify(LbaType lba, uint64_t /*timestamp*/) const {
    const auto found = history_.find(lba);
    if (found == history_.end() || found->second.interval_count < min_history_) {
        classified_unknown_++;
        return Temperature::UNKNOWN;
    }

    const double mean_interval =
        found->second.interval_sum / static_cast<double>(found->second.interval_count);
    if (mean_interval <= hot_threshold_) {
        classified_hot_++;
        return Temperature::HOT;
    }
    classified_cold_++;
    return Temperature::COLD;
}

void RuleBasedClassifier::observe(LbaType lba, uint64_t timestamp) {
    // try_emplace tells us whether this is the LBA's first write. Inferring it
    // from the stored timestamp would be wrong: normalized traces start at
    // timestamp 0, so a genuine first write can carry the same value as a
    // default-constructed entry.
    const auto result = history_.try_emplace(lba);
    History& entry = result.first->second;

    if (result.second) {
        // First sighting: no previous write, so no interval exists. Recording a
        // zero here would mark every newly written block as maximally hot.
        entry.last_write_timestamp = timestamp;
        return;
    }

    if (timestamp >= entry.last_write_timestamp) {
        entry.interval_sum += static_cast<double>(timestamp - entry.last_write_timestamp);
        entry.interval_count++;
    }
    entry.last_write_timestamp = timestamp;
}

} // namespace smartgc
