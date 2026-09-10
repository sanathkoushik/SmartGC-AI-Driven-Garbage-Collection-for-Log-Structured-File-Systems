// Tests for the pieces that connect the ML subsystem to the storage engine:
// the normalized-trace reader, the prediction merge-join, the rule-based
// control, and the end-to-end property that every policy sees an identical
// logical workload.

#include "test_framework.hpp"
#include "lfs_simulator.hpp"
#include "prediction_reader.hpp"
#include "rule_based_classifier.hpp"
#include "trace_reader.hpp"

#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace smartgc;

namespace {

std::string write_file(const std::string& name, const std::string& contents) {
    std::ofstream out(name);
    if (!out.is_open()) {
        throw std::runtime_error("test could not create " + name);
    }
    out << contents;
    out.close();
    return name;
}

struct TempFile {
    std::string path;
    explicit TempFile(const std::string& name, const std::string& contents)
        : path(write_file(name, contents)) {}
    ~TempFile() { std::remove(path.c_str()); }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
};

const char* kTraceCsv =
    "timestamp,lba,size,operation,trace_id\n"
    "0,10,1,W,demo\n"
    "5,11,1,W,demo\n"
    "5,12,1,W,demo\n"          // same timestamp, different LBA (one expanded request)
    "9,10,1,R,demo\n"          // a read: counted, but changes no state
    "12,10,1,W,demo\n"
    "20,11,1,W,demo\n"
    "31,10,1,W,demo\n";

} // namespace

// -----------------------------------------------------------------------------
TEST_CASE(test_trace_reader_parses_contract) {
    TempFile file("smartgc_test_trace.csv", kTraceCsv);
    TraceReader reader(file.path);

    std::vector<TraceRecord> records;
    TraceRecord record;
    while (reader.next(record)) {
        records.push_back(record);
    }

    ASSERT_EQ(records.size(), 7);
    ASSERT_EQ(reader.rows_read(), 7);
    ASSERT_EQ(reader.malformed_rows(), 0);
    ASSERT_EQ(reader.trace_id(), std::string("demo"));

    ASSERT_EQ(records[0].timestamp, 0);
    ASSERT_EQ(records[0].lba, 10);
    ASSERT_EQ(records[0].operation, 'W');
    ASSERT_EQ(records[2].timestamp, 5);
    ASSERT_EQ(records[2].lba, 12);
    ASSERT_EQ(records[3].operation, 'R');
    ASSERT_EQ(records[6].timestamp, 31);
}

// -----------------------------------------------------------------------------
TEST_CASE(test_trace_reader_rejects_bad_rows_loudly) {
    TempFile file("smartgc_test_badtrace.csv",
        "timestamp,lba,size,operation,trace_id\n"
        "0,10,1,W,demo\n"
        "not_a_number,11,1,W,demo\n"     // malformed timestamp
        "5,-3,1,W,demo\n"                // negative LBA
        "6,12,2,W,demo\n"                // size != 1: preprocessing must expand
        "7,13,1,X,demo\n"                // unknown operation
        "8,14,1,W,demo\n");
    TraceReader reader(file.path);

    size_t accepted = 0;
    TraceRecord record;
    while (reader.next(record)) {
        ++accepted;
    }

    ASSERT_EQ(accepted, 2);                       // only the two well-formed writes
    ASSERT_EQ(reader.rows_read(), 6);
    ASSERT_EQ(reader.malformed_rows(), 1);
    ASSERT_EQ(reader.negative_lba_rows(), 1);
    ASSERT_EQ(reader.non_unit_size_rows(), 1);
    ASSERT_EQ(reader.unknown_operation_rows(), 1);

    // A wrong header must fail immediately rather than mis-parse every row.
    TempFile bad_header("smartgc_test_badheader.csv", "a,b,c,d\n1,2,3,4\n");
    bool threw = false;
    try {
        TraceReader broken(bad_header.path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    ASSERT_TRUE(threw);

    bool missing_threw = false;
    try {
        TraceReader missing("definitely_not_a_file.csv");
    } catch (const std::runtime_error&) {
        missing_threw = true;
    }
    ASSERT_TRUE(missing_threw);
}

// -----------------------------------------------------------------------------
TEST_CASE(test_prediction_reader_merge_join) {
    TempFile file("smartgc_test_predictions.csv",
        "timestamp,lba,predicted_rewrite_interval,predicted_class,model_version,trace_id\n"
        "5,11,120.5,HOT,m1,demo\n"
        "5,12,900000.0,COLD,m1,demo\n"
        "12,10,50.0,HOT,m1,demo\n"
        "31,10,7.5,HOT,m1,demo\n");
    PredictionReader reader(file.path);

    double interval = 0.0;
    // The very first write has no prediction: not enough history yet.
    ASSERT_TRUE(reader.lookup(0, 10) == Temperature::UNKNOWN);
    // Two LBAs share a timestamp; both must resolve independently.
    ASSERT_TRUE(reader.lookup(5, 11, interval) == Temperature::HOT);
    ASSERT_NEAR(interval, 120.5, 1e-9);
    ASSERT_TRUE(reader.lookup(5, 12, interval) == Temperature::COLD);
    ASSERT_NEAR(interval, 900000.0, 1e-9);
    // An LBA that is absent from this timestamp's group must miss, not borrow.
    ASSERT_TRUE(reader.lookup(5, 99) == Temperature::UNKNOWN);
    ASSERT_TRUE(reader.lookup(12, 10) == Temperature::HOT);
    ASSERT_TRUE(reader.lookup(20, 11) == Temperature::UNKNOWN);
    ASSERT_TRUE(reader.lookup(31, 10) == Temperature::HOT);

    reader.finalize();
    ASSERT_EQ(reader.rows_read(), 4);
    ASSERT_EQ(reader.rows_matched(), 4);
    ASSERT_EQ(reader.lookups(), 7);
    ASSERT_EQ(reader.lookup_misses(), 3);
    ASSERT_EQ(reader.unmatched_predictions(), 0);
    ASSERT_EQ(reader.model_version(), std::string("m1"));
    ASSERT_FALSE(reader.out_of_order_input());
}

// -----------------------------------------------------------------------------
TEST_CASE(test_prediction_reader_reports_anomalies) {
    TempFile file("smartgc_test_badpredictions.csv",
        "timestamp,lba,predicted_rewrite_interval,predicted_class,model_version,trace_id\n"
        "1,10,5.0,HOT,m1,demo\n"
        "2,11,oops,HOT,m1,demo\n"        // malformed interval
        "3,12,5.0,LUKEWARM,m1,demo\n"    // unknown class: must not become COLD
        "4,13,5.0,COLD,m1,demo\n"
        "4,13,6.0,HOT,m1,demo\n"         // duplicate key
        "9,14,5.0,HOT,m1,demo\n");       // never queried
    PredictionReader reader(file.path);

    ASSERT_TRUE(reader.lookup(1, 10) == Temperature::HOT);
    ASSERT_TRUE(reader.lookup(2, 11) == Temperature::UNKNOWN);   // dropped as malformed
    ASSERT_TRUE(reader.lookup(3, 12) == Temperature::UNKNOWN);   // dropped as unknown label
    ASSERT_TRUE(reader.lookup(4, 13) == Temperature::COLD);      // first of the duplicates wins
    reader.finalize();

    ASSERT_EQ(reader.malformed_rows(), 1);
    ASSERT_EQ(reader.unknown_class_rows(), 1);
    ASSERT_EQ(reader.duplicate_key_rows(), 1);
    ASSERT_EQ(reader.unmatched_predictions(), 1);                // the row at timestamp 9
}

// -----------------------------------------------------------------------------
TEST_CASE(test_rule_based_classifier) {
    // Threshold 15 us, and an LBA must have 2 observed intervals before the
    // rule will commit to a temperature.
    RuleBasedClassifier rule(15.0, 2);

    // First write: no history at all.
    ASSERT_TRUE(rule.classify(10, 0) == Temperature::UNKNOWN);
    rule.observe(10, 0);
    // One interval so far: still below min_history.
    ASSERT_TRUE(rule.classify(10, 10) == Temperature::UNKNOWN);
    rule.observe(10, 10);
    // Two intervals (10, 10) -> mean 10 <= 15 -> HOT.
    ASSERT_TRUE(rule.classify(10, 20) == Temperature::UNKNOWN);
    rule.observe(10, 20);
    ASSERT_TRUE(rule.classify(10, 30) == Temperature::HOT);
    rule.observe(10, 30);

    // A slowly rewritten LBA: intervals 100, 100 -> mean 100 > 15 -> COLD.
    RuleBasedClassifier cold_rule(15.0, 2);
    cold_rule.observe(20, 0);
    cold_rule.observe(20, 100);
    cold_rule.observe(20, 200);
    ASSERT_TRUE(cold_rule.classify(20, 300) == Temperature::COLD);

    // An LBA first written at timestamp 0 must not be mistaken for "unseen" on
    // its second write: that would restart its history forever.
    RuleBasedClassifier zero_rule(1e9, 1);
    zero_rule.observe(7, 0);
    ASSERT_TRUE(zero_rule.classify(7, 0) == Temperature::UNKNOWN);
    zero_rule.observe(7, 5);
    ASSERT_TRUE(zero_rule.classify(7, 10) == Temperature::HOT);
    ASSERT_EQ(zero_rule.tracked_lbas(), 1);
}

// -----------------------------------------------------------------------------
TEST_CASE(test_policies_receive_identical_workload) {
    // The central fairness property: MIXED, RULE_BASED and LSTM_SMARTGC must
    // consume exactly the same logical request stream, so any WAF difference is
    // attributable to placement alone.
    TempFile trace("smartgc_test_fair_trace.csv", kTraceCsv);
    TempFile predictions("smartgc_test_fair_predictions.csv",
        "timestamp,lba,predicted_rewrite_interval,predicted_class,model_version,trace_id\n"
        "12,10,5.0,HOT,m1,demo\n"
        "20,11,900000.0,COLD,m1,demo\n"
        "31,10,5.0,HOT,m1,demo\n");

    auto replay = [&](PlacementPolicy policy, const std::string* prediction_path) {
        SimulatorConfig config;
        config.total_segments = 8;
        config.blocks_per_segment = 4;
        config.gc_free_segments_threshold = 1;
        config.gc_reserved_segments = 1;
        config.placement_policy = policy;

        LfsSimulator sim(config);
        std::unique_ptr<PredictionReader> reader;
        if (prediction_path) {
            reader = std::make_unique<PredictionReader>(*prediction_path);
        }
        RuleBasedClassifier rule(15.0, 1);

        TraceReader trace_reader(trace.path);
        TraceRecord record;
        size_t writes = 0;
        while (trace_reader.next(record)) {
            if (record.operation != 'W') {
                continue;
            }
            Temperature temperature = Temperature::UNKNOWN;
            if (policy == PlacementPolicy::RULE_BASED) {
                temperature = rule.classify(record.lba, record.timestamp);
                rule.observe(record.lba, record.timestamp);
            } else if (policy == PlacementPolicy::LSTM_SMARTGC && reader) {
                temperature = reader->lookup(record.timestamp, record.lba);
            }
            sim.write(record.lba, config.block_size_bytes, temperature);
            ++writes;
        }
        return std::make_pair(writes, sim.metrics());
    };

    const auto mixed = replay(PlacementPolicy::MIXED, nullptr);
    const auto rule_based = replay(PlacementPolicy::RULE_BASED, nullptr);
    const auto smart = replay(PlacementPolicy::LSTM_SMARTGC, &predictions.path);

    // Six writes in the fixture (the seventh row is a read).
    ASSERT_EQ(mixed.first, 6);
    ASSERT_EQ(rule_based.first, 6);
    ASSERT_EQ(smart.first, 6);

    // Identical logical workload for every policy: same request count and the
    // same logical bytes. Only physical bytes may differ.
    ASSERT_EQ(mixed.second.total_write_requests, rule_based.second.total_write_requests);
    ASSERT_EQ(mixed.second.total_write_requests, smart.second.total_write_requests);
    ASSERT_EQ(mixed.second.logical_bytes_written, rule_based.second.logical_bytes_written);
    ASSERT_EQ(mixed.second.logical_bytes_written, smart.second.logical_bytes_written);

    // MIXED never opens the cold stream; the prediction-driven policy does.
    ASSERT_EQ(mixed.second.cold_stream_writes, 0);
    ASSERT_EQ(mixed.second.unpredicted_writes, 6);
    ASSERT_TRUE(smart.second.cold_stream_writes > 0);
    // Three of the six writes had predictions in the fixture.
    ASSERT_EQ(smart.second.unpredicted_writes, 3);
}

// -----------------------------------------------------------------------------
TEST_CASE(test_replay_preserves_validity_semantics) {
    // Temperature drives placement only: after replaying a trace, every LBA the
    // workload wrote must still be live, whatever it was labelled.
    TempFile trace("smartgc_test_validity_trace.csv", kTraceCsv);

    SimulatorConfig config;
    config.total_segments = 8;
    config.blocks_per_segment = 4;
    config.gc_free_segments_threshold = 1;
    config.gc_reserved_segments = 1;
    config.placement_policy = PlacementPolicy::LSTM_SMARTGC;

    LfsSimulator sim(config);
    TraceReader reader(trace.path);
    TraceRecord record;
    size_t index = 0;
    while (reader.next(record)) {
        if (record.operation != 'W') {
            continue;
        }
        // Alternate the label deliberately: mislabelling must never lose data.
        const Temperature temperature = (index++ % 2 == 0) ? Temperature::HOT : Temperature::COLD;
        sim.write(record.lba, config.block_size_bytes, temperature);
    }

    ASSERT_TRUE(sim.read(10));
    ASSERT_TRUE(sim.read(11));
    ASSERT_TRUE(sim.read(12));
    ASSERT_EQ(sim.live_blocks(), 3);
    ASSERT_EQ(sim.count_total_valid_blocks(), sim.mapping().size());
    ASSERT_EQ(sim.metrics().physical_bytes_written,
              sim.metrics().logical_bytes_written + sim.metrics().gc_bytes_copied);
}
