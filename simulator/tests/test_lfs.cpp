#include "test_framework.hpp"
#include "types.hpp"
#include "block.hpp"
#include "segment.hpp"
#include "mapping.hpp"
#include "metrics.hpp"
#include "config.hpp"
#include "lfs_simulator.hpp"
#include "synthetic_workload.hpp"

#include <cstdio>
#include <fstream>
#include <set>
#include <string>

using namespace smartgc;

namespace {

// Re-checks every structural invariant the engine promises. Called repeatedly
// inside the longer workloads so a violation is caught at the request that
// introduced it rather than at the end of the run.
void assert_invariants(const LfsSimulator& sim) {
    // 1. Mapped LBAs and VALID blocks are the same set, counted two ways.
    ASSERT_EQ(sim.count_total_valid_blocks(), sim.mapping().size());

    // 2. Every mapping resolves to a VALID block holding exactly that LBA.
    for (const auto& entry : sim.mapping().table()) {
        const Segment& seg = sim.segments()[entry.second.segment_id];
        const Block& blk = seg.get_block(entry.second.block_index);
        ASSERT_TRUE(blk.is_valid());
        ASSERT_EQ(blk.lba, entry.first);
    }

    // 3. Physical writes are exactly user writes plus migrations.
    const StorageMetrics& m = sim.metrics();
    ASSERT_EQ(m.physical_bytes_written, m.logical_bytes_written + m.gc_bytes_copied);

    // 4. Per-segment counters agree with the blocks they describe.
    for (const Segment& seg : sim.segments()) {
        size_t valid = 0, invalid = 0, free_blocks = 0;
        for (size_t b = 0; b < seg.capacity(); ++b) {
            const Block& blk = seg.get_block(b);
            if (blk.is_valid()) ++valid;
            else if (blk.is_invalid()) ++invalid;
            else ++free_blocks;
        }
        ASSERT_EQ(seg.valid_count(), valid);
        ASSERT_EQ(seg.invalid_count(), invalid);
        ASSERT_EQ(seg.free_count(), free_blocks);
    }

    // 5. Live data never exceeds the configured usable capacity.
    ASSERT_TRUE(sim.live_blocks() <= sim.config().max_live_blocks());
}

SimulatorConfig small_config(size_t segments, size_t blocks_per_segment) {
    SimulatorConfig c;
    c.total_segments = segments;
    c.blocks_per_segment = blocks_per_segment;
    c.block_size_bytes = 4096;
    c.gc_free_segments_threshold = 1;
    c.gc_reserved_segments = 1;
    c.gc_separate_stream = true;
    c.placement_policy = PlacementPolicy::MIXED;
    return c;
}

std::string write_temp_file(const std::string& name, const std::string& contents) {
    std::ofstream out(name);
    if (!out.is_open()) {
        throw std::runtime_error("test could not create temp file " + name);
    }
    out << contents;
    out.close();
    return name;
}

bool throws_runtime(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

} // namespace

// -----------------------------------------------------------------------------
// Test 1: Block State Transitions
// -----------------------------------------------------------------------------
TEST_CASE(test_block_state_transitions) {
    Block b;
    ASSERT_TRUE(b.is_free());
    ASSERT_FALSE(b.is_valid());
    ASSERT_FALSE(b.is_invalid());
    ASSERT_EQ(b.lba, INVALID_LBA);

    b.allocate(100);
    ASSERT_FALSE(b.is_free());
    ASSERT_TRUE(b.is_valid());
    ASSERT_EQ(b.lba, 100);

    b.invalidate();
    ASSERT_TRUE(b.is_invalid());
    ASSERT_EQ(b.lba, 100);

    b.reset();
    ASSERT_TRUE(b.is_free());
    ASSERT_EQ(b.lba, INVALID_LBA);
}

// -----------------------------------------------------------------------------
// Test 2: Segment Capacity, Allocation, Invalidation, Reset
// -----------------------------------------------------------------------------
TEST_CASE(test_segment_operations) {
    Segment seg(0, 4);
    ASSERT_EQ(seg.id(), 0);
    ASSERT_EQ(seg.capacity(), 4);
    ASSERT_EQ(seg.free_count(), 4);
    ASSERT_TRUE(seg.has_free_block());
    ASSERT_FALSE(seg.is_full());

    ASSERT_EQ(seg.allocate_block(10), 0);
    ASSERT_EQ(seg.allocate_block(20), 1);
    ASSERT_EQ(seg.allocate_block(30), 2);
    ASSERT_EQ(seg.allocate_block(40), 3);

    ASSERT_EQ(seg.valid_count(), 4);
    ASSERT_EQ(seg.free_count(), 0);
    ASSERT_TRUE(seg.is_full());
    ASSERT_FALSE(seg.has_free_block());

    seg.invalidate_block(1);
    ASSERT_EQ(seg.valid_count(), 3);
    ASSERT_EQ(seg.invalid_count(), 1);
    ASSERT_NEAR(seg.valid_ratio(), 0.75, 1e-9);
    ASSERT_NEAR(seg.invalid_ratio(), 0.25, 1e-9);

    // Invalidating the same block twice must not double-count.
    seg.invalidate_block(1);
    ASSERT_EQ(seg.valid_count(), 3);
    ASSERT_EQ(seg.invalid_count(), 1);

    seg.reset();
    ASSERT_EQ(seg.valid_count(), 0);
    ASSERT_EQ(seg.invalid_count(), 0);
    ASSERT_EQ(seg.free_count(), 4);
    ASSERT_TRUE(seg.has_free_block());
}

// -----------------------------------------------------------------------------
// Test 3: Out-of-Place Overwrite and L2P Remapping
// -----------------------------------------------------------------------------
TEST_CASE(test_lba_mapping_and_overwrites) {
    LfsSimulator sim(small_config(6, 4));

    sim.write(5);
    ASSERT_TRUE(sim.read(5));
    ASSERT_FALSE(sim.read(99));

    PhysicalAddress addr1;
    ASSERT_TRUE(sim.mapping().get(5, addr1));
    ASSERT_EQ(addr1.block_index, 0);
    ASSERT_TRUE(sim.segments()[addr1.segment_id].get_block(0).is_valid());

    sim.write(5);
    PhysicalAddress addr2;
    ASSERT_TRUE(sim.mapping().get(5, addr2));
    ASSERT_EQ(addr2.segment_id, addr1.segment_id);
    ASSERT_EQ(addr2.block_index, 1);

    ASSERT_TRUE(sim.segments()[addr1.segment_id].get_block(0).is_invalid());
    ASSERT_TRUE(sim.segments()[addr2.segment_id].get_block(1).is_valid());

    const StorageMetrics& m = sim.metrics();
    ASSERT_EQ(m.total_write_requests, 2);
    ASSERT_EQ(m.logical_bytes_written, 2u * 4096u);
    ASSERT_EQ(m.physical_bytes_written, 2u * 4096u);
    ASSERT_EQ(m.gc_bytes_copied, 0);
    ASSERT_EQ(m.gc_count, 0);
    ASSERT_NEAR(m.waf(), 1.0, 1e-12);
    // One LBA, written twice: exactly one block is live.
    ASSERT_EQ(sim.live_blocks(), 1);
    assert_invariants(sim);
}

// -----------------------------------------------------------------------------
// Test 4: Greedy Victim Selection Picks the Fewest Valid Blocks
// -----------------------------------------------------------------------------
TEST_CASE(test_greedy_gc_victim_selection) {
    LfsSimulator sim(small_config(8, 4));

    for (int i = 0; i < 4; ++i) sim.write(i);         // segment A: LBAs 0..3
    for (int i = 4; i < 8; ++i) sim.write(i);         // segment B: LBAs 4..7
    for (int i = 8; i < 12; ++i) sim.write(i);        // segment C: LBAs 8..11

    // Invalidate 3 of segment A and 1 of segment B.
    sim.write(0); sim.write(1); sim.write(2);
    sim.write(4);

    const size_t seg_a = 0, seg_b = 1, seg_c = 2;
    ASSERT_EQ(sim.segments()[seg_a].valid_count(), 1);   // only LBA 3 survives
    ASSERT_EQ(sim.segments()[seg_b].valid_count(), 3);
    ASSERT_EQ(sim.segments()[seg_c].valid_count(), 4);
    assert_invariants(sim);

    ASSERT_TRUE(sim.run_greedy_gc());

    // Segment A had the fewest valid blocks and must have been cleaned.
    ASSERT_EQ(sim.segments()[seg_a].valid_count(), 0);
    ASSERT_EQ(sim.segments()[seg_a].invalid_count(), 0);
    ASSERT_EQ(sim.segments()[seg_a].free_count(), 4);

    // Its single survivor was relocated, not lost.
    ASSERT_TRUE(sim.read(3));
    PhysicalAddress addr;
    ASSERT_TRUE(sim.mapping().get(3, addr));
    ASSERT_NE(addr.segment_id, seg_a);
    ASSERT_EQ(sim.metrics().valid_blocks_migrated, 1);
    ASSERT_EQ(sim.metrics().gc_count, 1);
    assert_invariants(sim);
}

// -----------------------------------------------------------------------------
// Test 5: A fully valid segment is never chosen as a victim
// -----------------------------------------------------------------------------
TEST_CASE(test_gc_skips_fully_valid_segments) {
    LfsSimulator sim(small_config(8, 4));

    // Two closed segments, both entirely valid.
    for (int i = 0; i < 8; ++i) sim.write(i);
    // Third segment, partially invalidated by overwriting LBAs 0 and 1.
    sim.write(0);
    sim.write(1);
    sim.write(100);
    sim.write(101);

    // Segment 0 now has 2 valid, 2 invalid; segment 1 is fully valid.
    ASSERT_EQ(sim.segments()[0].valid_count(), 2);
    ASSERT_EQ(sim.segments()[1].valid_count(), 4);

    ASSERT_TRUE(sim.run_greedy_gc());
    // The fully valid segment 1 must be untouched.
    ASSERT_EQ(sim.segments()[1].valid_count(), 4);
    ASSERT_EQ(sim.segments()[0].free_count(), 4);   // segment 0 was reclaimed
    assert_invariants(sim);
}

// -----------------------------------------------------------------------------
// Test 6: Hand-Computable Deterministic WAF Verification
// -----------------------------------------------------------------------------
TEST_CASE(test_hand_computed_waf) {
    // -------------------------------------------------------------------------
    // Geometry: 6 segments x 4 blocks, 4096-byte blocks, GC watermark 1,
    // GC reserve 1, separate GC stream, MIXED placement.
    //   unavailable segments = reserve(1) + user append point(1) + GC append(1) = 3
    //   max live blocks       = (6 - 3) * 4 = 12
    //   cleaning floor        = max(watermark 1, reserve 1) = 1
    // The free pool is FIFO: allocation pops the front, reclamation pushes back.
    //
    // Write sequence (21 user writes):
    //   0,1,2,3  4,5,6,7  8,9,10,11  0,1,2  4  5  6  0  1  2
    //
    // Step-by-step:
    //  w1-w4   LBAs 0..3      -> pop seg0; seg0 = [L0 L1 L2 L3], full. free=[1..5]
    //  w5-w8   LBAs 4..7      -> close seg0, pop seg1; seg1 = [L4 L5 L6 L7]. free=[2..5]
    //  w9-w12  LBAs 8..11     -> close seg1, pop seg2; seg2 = [L8 L9 L10 L11]. free=[3,4,5]
    //                            live = 12 blocks = the usable maximum.
    //  w13-w15 LBAs 0,1,2     -> close seg2, pop seg3; seg3 = [L0 L1 L2 _ ]. free=[4,5]
    //                            seg0 -> valid 1 (L3), invalid 3
    //  w16     LBA 4          -> seg3[3] = L4, seg3 full.
    //                            seg1 -> valid 3 (L5 L6 L7), invalid 1
    //  w17     LBA 5          -> close seg3; free=2 > floor 1, no cleaning;
    //                            pop seg4; seg4[0] = L5. free=[5]
    //                            seg1 -> valid 2 (L6 L7), invalid 2
    //  w18     LBA 6          -> seg4[1] = L6. seg1 -> valid 1 (L7), invalid 3
    //  w19     LBA 0          -> seg4[2] = L0. seg3 -> valid 3, invalid 1
    //  w20     LBA 1          -> seg4[3] = L1, seg4 full. seg3 -> valid 2 (L2 L4), invalid 2
    //
    //  State before the final write:
    //    seg0 valid 1 (L3)   seg1 valid 1 (L7)   seg2 valid 4 (FULL - never a victim)
    //    seg3 valid 2        seg4 valid 4 (FULL) seg5 free.   free pool = [5], size 1
    //
    //  w21     LBA 2          -> close seg4. free = 1 <= floor 1, so clean:
    //      pass 1: candidates seg0(v1,i3), seg1(v1,i3), seg3(v2,i2);
    //              seg2 and seg4 are fully valid and are skipped.
    //              greedy picks seg0 (1 valid, tie broken by first-seen).
    //              migrate L3 -> GC stream opens seg5 (free -> 0), seg5[0] = L3.
    //                +1 physical, +1 gc_copied, +1 migrated.
    //              reclaim seg0 -> free = [0], size 1.
    //      free = 1 <= 1, so clean again:
    //      pass 2: candidates seg1(v1,i3), seg3(v2,i2); seg5 is the open GC
    //              stream and is skipped. greedy picks seg1.
    //              migrate L7 -> seg5 still has room, seg5[1] = L7.
    //                +1 physical, +1 gc_copied, +1 migrated.
    //              reclaim seg1 -> free = [0,1], size 2 > 1, cleaning stops.
    //      pop seg0 for MAIN; invalidate L2 in seg3; seg0[0] = L2.
    //
    // Totals:
    //   user write requests    = 21
    //   logical bytes          = 21 * 4096 = 86,016
    //   valid blocks migrated  = 2  -> gc bytes = 2 * 4096 = 8,192
    //   physical bytes         = (21 + 2) * 4096 = 23 * 4096 = 94,208
    //   GC invocations         = 2
    //   WAF                    = 23 / 21 = 1.095238095238...
    // -------------------------------------------------------------------------

    SimulatorConfig config = small_config(6, 4);
    ASSERT_EQ(config.max_live_blocks(), 12);

    LfsSimulator sim(config);

    const LbaType sequence[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                0, 1, 2, 4, 5, 6, 0, 1, 2};
    for (LbaType lba : sequence) {
        sim.write(lba);
    }

    const StorageMetrics& m = sim.metrics();
    ASSERT_EQ(m.total_write_requests, 21);
    ASSERT_EQ(m.logical_bytes_written, 21u * 4096u);
    ASSERT_EQ(m.valid_blocks_migrated, 2);
    ASSERT_EQ(m.gc_bytes_copied, 2u * 4096u);
    ASSERT_EQ(m.physical_bytes_written, 23u * 4096u);
    ASSERT_EQ(m.gc_count, 2);
    ASSERT_NEAR(m.waf(), 23.0 / 21.0, 1e-12);

    // Every LBA ever written must still be readable: cleaning relocates data,
    // it never drops it.
    for (LbaType lba = 0; lba <= 11; ++lba) {
        ASSERT_TRUE(sim.read(lba));
    }
    ASSERT_EQ(sim.live_blocks(), 12);
    assert_invariants(sim);
}

// -----------------------------------------------------------------------------
// Test 7: High utilization must not exhaust the free pool (regression)
// -----------------------------------------------------------------------------
TEST_CASE(test_high_utilization_completes_without_starvation) {
    // Before the GC reserve was introduced, a skewed workload at this
    // utilization aborted mid-run with "No free segment available to receive
    // migrated valid block". It must now complete and stay consistent.
    SimulatorConfig config;
    config.total_segments = 32;
    config.blocks_per_segment = 64;
    config.gc_free_segments_threshold = 2;
    config.gc_reserved_segments = 2;

    const size_t working_set = config.max_live_blocks();   // the usable maximum
    ASSERT_EQ(working_set, 28u * 64u);

    LfsSimulator sim(config);
    const std::vector<WorkloadRequest> requests =
        SyntheticWorkloadGenerator::generate_skewed(
            60000, static_cast<LbaType>(working_set), 0.2, 0.8, 1234);

    for (size_t i = 0; i < requests.size(); ++i) {
        sim.write(requests[i].lba);
        if (i % 5000 == 0) {
            assert_invariants(sim);
            // The reserve must never be consumed by user writes.
            ASSERT_TRUE(sim.free_segment_count() >= 1);
        }
    }
    assert_invariants(sim);

    ASSERT_EQ(sim.metrics().total_write_requests, 60000);
    // At this utilization cleaning is unavoidable, so WAF must exceed 1.
    ASSERT_TRUE(sim.metrics().gc_count > 0);
    ASSERT_TRUE(sim.metrics().waf() > 1.0);
}

// -----------------------------------------------------------------------------
// Test 8: Admission control rejects a working set larger than usable capacity
// -----------------------------------------------------------------------------
TEST_CASE(test_device_full_admission_control) {
    SimulatorConfig config = small_config(6, 4);
    ASSERT_EQ(config.max_live_blocks(), 12);

    LfsSimulator sim(config);
    for (LbaType lba = 0; lba < 12; ++lba) {
        sim.write(lba);
    }
    ASSERT_EQ(sim.live_blocks(), 12);

    // The 13th distinct LBA cannot be admitted...
    bool threw_device_full = false;
    try {
        sim.write(12);
    } catch (const DeviceFullError&) {
        threw_device_full = true;
    }
    ASSERT_TRUE(threw_device_full);

    // ...but overwriting an existing LBA is always accepted, because it does not
    // increase the live set.
    ASSERT_TRUE(sim.write(0));
    assert_invariants(sim);
}

// -----------------------------------------------------------------------------
// Test 9: Multi-block requests must not be silently truncated to one block
// -----------------------------------------------------------------------------
TEST_CASE(test_write_rejects_non_block_sized_request) {
    LfsSimulator sim(small_config(6, 4));

    bool threw = false;
    try {
        sim.write(1, 8192);              // two blocks in a single call
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
    // The rejected request must leave no trace in the metrics.
    ASSERT_EQ(sim.metrics().total_write_requests, 0);
    ASSERT_EQ(sim.metrics().logical_bytes_written, 0);

    ASSERT_TRUE(sim.write(1, 4096));
    ASSERT_EQ(sim.metrics().total_write_requests, 1);

    // A negative LBA is rejected without being counted either.
    ASSERT_FALSE(sim.write(-1, 4096));
    ASSERT_EQ(sim.metrics().total_write_requests, 1);
}

// -----------------------------------------------------------------------------
// Test 10: GC migrations use their own append point when configured
// -----------------------------------------------------------------------------
TEST_CASE(test_gc_stream_separation) {
    SimulatorConfig config = small_config(8, 4);
    config.gc_separate_stream = true;

    LfsSimulator sim(config);
    for (int i = 0; i < 8; ++i) sim.write(i);
    sim.write(0); sim.write(1); sim.write(2);       // segment 0 -> 1 valid (LBA 3)

    ASSERT_EQ(sim.open_segment_id(WriteStream::GC), static_cast<size_t>(-1));
    ASSERT_TRUE(sim.run_greedy_gc());

    const size_t gc_seg = sim.open_segment_id(WriteStream::GC);
    const size_t main_seg = sim.open_segment_id(WriteStream::MAIN);
    ASSERT_NE(gc_seg, static_cast<size_t>(-1));
    ASSERT_NE(gc_seg, main_seg);

    // The migrated survivor landed in the GC stream, not the user log.
    PhysicalAddress addr;
    ASSERT_TRUE(sim.mapping().get(3, addr));
    ASSERT_EQ(addr.segment_id, gc_seg);

    // With a shared stream the survivor follows the user log instead.
    SimulatorConfig shared = small_config(8, 4);
    shared.gc_separate_stream = false;
    LfsSimulator sim2(shared);
    for (int i = 0; i < 8; ++i) sim2.write(i);
    sim2.write(0); sim2.write(1); sim2.write(2);
    ASSERT_TRUE(sim2.run_greedy_gc());
    ASSERT_EQ(sim2.open_segment_id(WriteStream::GC), static_cast<size_t>(-1));
    PhysicalAddress addr2;
    ASSERT_TRUE(sim2.mapping().get(3, addr2));
    ASSERT_EQ(addr2.segment_id, sim2.open_segment_id(WriteStream::MAIN));
}

// -----------------------------------------------------------------------------
// Test 11: Temperature routes placement, and never affects validity
// -----------------------------------------------------------------------------
TEST_CASE(test_temperature_routes_placement_only) {
    SimulatorConfig config = small_config(8, 4);
    config.placement_policy = PlacementPolicy::RULE_BASED;

    LfsSimulator sim(config);
    sim.write(10, 4096, Temperature::HOT);
    sim.write(20, 4096, Temperature::COLD);
    sim.write(11, 4096, Temperature::HOT);
    sim.write(21, 4096, Temperature::COLD);

    const size_t main_seg = sim.open_segment_id(WriteStream::MAIN);
    const size_t cold_seg = sim.open_segment_id(WriteStream::COLD);
    ASSERT_NE(main_seg, cold_seg);

    PhysicalAddress a;
    ASSERT_TRUE(sim.mapping().get(10, a)); ASSERT_EQ(a.segment_id, main_seg);
    ASSERT_TRUE(sim.mapping().get(11, a)); ASSERT_EQ(a.segment_id, main_seg);
    ASSERT_TRUE(sim.mapping().get(20, a)); ASSERT_EQ(a.segment_id, cold_seg);
    ASSERT_TRUE(sim.mapping().get(21, a)); ASSERT_EQ(a.segment_id, cold_seg);

    ASSERT_EQ(sim.metrics().main_stream_writes, 2);
    ASSERT_EQ(sim.metrics().cold_stream_writes, 2);
    ASSERT_EQ(sim.metrics().unpredicted_writes, 0);

    // Temperature is a placement hint only: a COLD-labelled block is still made
    // INVALID by the next overwrite, exactly like a HOT one.
    PhysicalAddress before;
    ASSERT_TRUE(sim.mapping().get(20, before));
    sim.write(20, 4096, Temperature::HOT);
    ASSERT_TRUE(sim.segments()[before.segment_id].get_block(before.block_index).is_invalid());
    ASSERT_EQ(sim.metrics().main_stream_writes, 3);
    assert_invariants(sim);

    // Under MIXED the cold append point is never opened, whatever the hint says.
    SimulatorConfig mixed = small_config(8, 4);
    LfsSimulator sim3(mixed);
    sim3.write(10, 4096, Temperature::COLD);
    sim3.write(11, 4096, Temperature::HOT);
    ASSERT_EQ(sim3.open_segment_id(WriteStream::COLD), static_cast<size_t>(-1));
    ASSERT_EQ(sim3.metrics().cold_stream_writes, 0);
    ASSERT_EQ(sim3.metrics().main_stream_writes, 2);

    // A write with no prediction is counted, and falls back to the main log.
    LfsSimulator sim4(config);
    sim4.write(30, 4096, Temperature::UNKNOWN);
    ASSERT_EQ(sim4.metrics().unpredicted_writes, 1);
    ASSERT_EQ(sim4.metrics().main_stream_writes, 1);
}

// -----------------------------------------------------------------------------
// Test 12: Configuration validation
// -----------------------------------------------------------------------------
TEST_CASE(test_config_validation) {
    SimulatorConfig zero_reserve = small_config(8, 4);
    zero_reserve.gc_reserved_segments = 0;
    ASSERT_TRUE(throws_runtime([&] { zero_reserve.validate(); }));

    SimulatorConfig too_small;
    too_small.total_segments = 3;          // reserve 2 + user 1 + gc 1 = 4 needed
    too_small.blocks_per_segment = 4;
    ASSERT_TRUE(throws_runtime([&] { too_small.validate(); }));

    SimulatorConfig no_blocks = small_config(8, 0);
    ASSERT_TRUE(throws_runtime([&] { no_blocks.validate(); }));

    SimulatorConfig ok = small_config(8, 4);
    ok.validate();                          // must not throw
    ASSERT_EQ(ok.total_capacity_blocks(), 32);
    ASSERT_EQ(ok.unavailable_segments(), 3);
    ASSERT_EQ(ok.max_live_blocks(), 20);
    ASSERT_NEAR(ok.max_utilization(), 20.0 / 32.0, 1e-12);

    // A segregated policy needs a second user append point, which reduces the
    // usable capacity by one segment.
    SimulatorConfig segregated = ok;
    segregated.placement_policy = PlacementPolicy::LSTM_SMARTGC;
    ASSERT_EQ(segregated.user_stream_count(), 2);
    ASSERT_EQ(segregated.max_live_blocks(), 16);

    // The constructor validates, so an invalid configuration cannot be run.
    ASSERT_TRUE(throws_runtime([&] { LfsSimulator bad(zero_reserve); (void)bad; }));
}

// -----------------------------------------------------------------------------
// Test 13: config.yaml loading
// -----------------------------------------------------------------------------
TEST_CASE(test_config_yaml_loading) {
    const std::string path = write_temp_file("smartgc_test_config.yaml",
        "# SmartGC test config\n"
        "random_seed: 7\n"
        "\n"
        "simulator:\n"
        "  block_size_bytes: 8192       # inline comment\n"
        "  blocks_per_segment: 16\n"
        "  total_segments: 40\n"
        "  gc_free_segments_threshold: 3\n"
        "  gc_reserved_segments: 4\n"
        "  gc_separate_stream: false\n"
        "  gc_policy: \"GREEDY\"\n"
        "  placement_policy: \"LSTM_SMARTGC\"\n"
        "\n"
        "synthetic_workload:\n"
        "  total_requests: 1234\n"
        "  lba_range_min: 0\n"
        "  lba_range_max: 321\n"
        "  hot_ratio: 0.25\n"
        "  hot_traffic_ratio: 0.9\n"
        "\n"
        "ml:\n"
        "  sequence_length: 10\n");

    const SimulatorConfig cfg = load_simulator_config(path);
    ASSERT_EQ(cfg.random_seed, 7);
    ASSERT_EQ(cfg.block_size_bytes, 8192);
    ASSERT_EQ(cfg.blocks_per_segment, 16);
    ASSERT_EQ(cfg.total_segments, 40);
    ASSERT_EQ(cfg.gc_free_segments_threshold, 3);
    ASSERT_EQ(cfg.gc_reserved_segments, 4);
    ASSERT_FALSE(cfg.gc_separate_stream);
    ASSERT_TRUE(cfg.placement_policy == PlacementPolicy::LSTM_SMARTGC);

    const WorkloadConfig wl = load_workload_config(path);
    ASSERT_EQ(wl.total_requests, 1234);
    ASSERT_EQ(wl.lba_range_max, 321);
    ASSERT_NEAR(wl.hot_ratio, 0.25, 1e-12);
    ASSERT_NEAR(wl.hot_traffic_ratio, 0.9, 1e-12);
    std::remove(path.c_str());

    // An unrecognised key must be an error, never a silent default.
    const std::string typo = write_temp_file("smartgc_test_typo.yaml",
        "simulator:\n  total_segmets: 40\n");
    ASSERT_TRUE(throws_runtime([&] { load_simulator_config(typo); }));
    std::remove(typo.c_str());

    // So must a malformed value, a missing section, and an absent file.
    const std::string bad_value = write_temp_file("smartgc_test_badvalue.yaml",
        "simulator:\n  total_segments: many\n");
    ASSERT_TRUE(throws_runtime([&] { load_simulator_config(bad_value); }));
    std::remove(bad_value.c_str());

    const std::string no_section = write_temp_file("smartgc_test_nosection.yaml",
        "ml:\n  sequence_length: 10\n");
    ASSERT_TRUE(throws_runtime([&] { load_simulator_config(no_section); }));
    std::remove(no_section.c_str());

    ASSERT_TRUE(throws_runtime([&] { load_simulator_config("does_not_exist.yaml"); }));
}

// -----------------------------------------------------------------------------
// Test 14: The repository's own config.yaml is loadable and valid
// -----------------------------------------------------------------------------
TEST_CASE(test_repository_config_is_valid) {
    // Located relative to the repository root; the test runner is invoked from
    // there. Skipped rather than failed when run from elsewhere, so the suite
    // stays usable from an arbitrary working directory.
    const char* path = "config/config.yaml";
    std::ifstream probe(path);
    if (!probe.good()) {
        std::cout << "             (skipped: config/config.yaml not reachable from CWD)\n";
        return;
    }
    probe.close();

    const SimulatorConfig cfg = load_simulator_config(path);
    cfg.validate();
    const WorkloadConfig wl = load_workload_config(path);
    wl.validate();

    // The shipped synthetic workload must fit the shipped geometry, otherwise
    // the default demo run would abort on a device-full error.
    ASSERT_TRUE(static_cast<size_t>(wl.lba_range_max) <= cfg.max_live_blocks());
}

// -----------------------------------------------------------------------------
// Test 15: Cleaning migrates valid blocks only, and preserves all live data
// -----------------------------------------------------------------------------
TEST_CASE(test_gc_migrates_only_valid_blocks) {
    SimulatorConfig config = small_config(10, 4);
    LfsSimulator sim(config);

    std::set<LbaType> written;
    const std::vector<WorkloadRequest> requests =
        SyntheticWorkloadGenerator::generate_skewed(2000, 20, 0.3, 0.8, 99);
    for (const WorkloadRequest& r : requests) {
        sim.write(r.lba);
        written.insert(r.lba);
    }
    assert_invariants(sim);

    // Every LBA the workload ever touched is still live and readable.
    ASSERT_EQ(sim.live_blocks(), written.size());
    for (LbaType lba : written) {
        ASSERT_TRUE(sim.read(lba));
    }

    // Migration accounting is consistent with the byte counters.
    const StorageMetrics& m = sim.metrics();
    ASSERT_EQ(m.gc_bytes_copied, m.valid_blocks_migrated * 4096u);
    ASSERT_EQ(m.logical_bytes_written, m.total_write_requests * 4096u);
    ASSERT_TRUE(m.gc_count > 0);

    // No segment can report more valid blocks than it has capacity for.
    for (const Segment& seg : sim.segments()) {
        ASSERT_TRUE(seg.valid_count() + seg.invalid_count() + seg.free_count() == seg.capacity());
    }
}

// -----------------------------------------------------------------------------
// Test 16: Determinism - identical inputs produce identical metrics
// -----------------------------------------------------------------------------
TEST_CASE(test_run_is_deterministic) {
    SimulatorConfig config = small_config(16, 8);
    const std::vector<WorkloadRequest> requests =
        SyntheticWorkloadGenerator::generate_skewed(5000, 64, 0.2, 0.8, 2024);

    auto run = [&]() {
        LfsSimulator sim(config);
        for (const WorkloadRequest& r : requests) sim.write(r.lba);
        return sim.metrics();
    };

    const StorageMetrics a = run();
    const StorageMetrics b = run();
    ASSERT_EQ(a.physical_bytes_written, b.physical_bytes_written);
    ASSERT_EQ(a.gc_bytes_copied, b.gc_bytes_copied);
    ASSERT_EQ(a.valid_blocks_migrated, b.valid_blocks_migrated);
    ASSERT_EQ(a.gc_count, b.gc_count);
    ASSERT_NEAR(a.waf(), b.waf(), 0.0);

    // The generator itself must be reproducible from its seed.
    const std::vector<WorkloadRequest> again =
        SyntheticWorkloadGenerator::generate_skewed(5000, 64, 0.2, 0.8, 2024);
    ASSERT_EQ(again.size(), requests.size());
    for (size_t i = 0; i < again.size(); ++i) {
        ASSERT_EQ(again[i].lba, requests[i].lba);
        ASSERT_EQ(again[i].size_blocks, 1u);
    }
}

// -----------------------------------------------------------------------------
// Test 17: Metrics CSV row matches its header
// -----------------------------------------------------------------------------
TEST_CASE(test_metrics_csv_contract) {
    auto count_fields = [](const std::string& s) {
        size_t n = 1;
        for (char c : s) if (c == ',') ++n;
        return n;
    };

    StorageMetrics m;
    m.logical_bytes_written = 4096;
    m.physical_bytes_written = 8192;
    m.gc_bytes_copied = 4096;
    m.valid_blocks_migrated = 1;
    m.gc_count = 1;
    m.total_write_requests = 1;

    RunDescriptor run;
    run.dataset = "msr";
    run.trace = "hm_0";
    run.policy = placement_policy_to_string(PlacementPolicy::LSTM_SMARTGC);
    run.model_type = "pretrained_finetuned";

    const std::string header = StorageMetrics::csv_header();
    const std::string row = run.to_csv_row(m);
    ASSERT_EQ(count_fields(header), count_fields(row));

    // The policy column must reflect the actual policy, not a hardcoded value.
    ASSERT_TRUE(row.find("LSTM_SMARTGC") != std::string::npos);
    ASSERT_NEAR(m.waf(), 2.0, 1e-12);
}

// -----------------------------------------------------------------------------
// Test 18: Shared GC stream must not strand the user append point (regression)
// -----------------------------------------------------------------------------
TEST_CASE(test_shared_gc_stream_does_not_strand_segments) {
    // With gc_separate_stream = false a cleaning pass migrates into the *user*
    // append point. If the user write path then opened a second segment instead
    // of adopting that one, the first would stay flagged open forever, drop out
    // of victim selection, and the device would report itself full after a few
    // hundred writes even at 24% utilization.
    SimulatorConfig config;
    config.total_segments = 32;
    config.blocks_per_segment = 64;
    config.gc_free_segments_threshold = 2;
    config.gc_reserved_segments = 2;
    config.gc_separate_stream = false;

    LfsSimulator sim(config);
    const std::vector<WorkloadRequest> requests =
        SyntheticWorkloadGenerator::generate_skewed(50000, 500, 0.2, 0.8, 42);

    for (size_t i = 0; i < requests.size(); ++i) {
        sim.write(requests[i].lba);
        if (i % 5000 == 0) {
            assert_invariants(sim);
            // Segments must keep circulating: at most the reserve plus the two
            // possible append points can be unavailable at any instant.
            size_t open_count = 0;
            for (const Segment& seg : sim.segments()) {
                if (seg.is_open()) ++open_count;
            }
            ASSERT_TRUE(open_count <= NUM_WRITE_STREAMS);
        }
    }
    assert_invariants(sim);
    ASSERT_EQ(sim.metrics().total_write_requests, 50000);
    ASSERT_TRUE(sim.metrics().gc_count > 0);

    // A shared stream never opens the dedicated GC append point.
    ASSERT_EQ(sim.open_segment_id(WriteStream::GC), static_cast<size_t>(-1));

    // The same workload must also complete with the default separate stream.
    SimulatorConfig separate = config;
    separate.gc_separate_stream = true;
    LfsSimulator sim2(separate);
    for (const WorkloadRequest& r : requests) sim2.write(r.lba);
    assert_invariants(sim2);
    ASSERT_EQ(sim2.metrics().total_write_requests, 50000);
}

int main() {
    return smartgc::test::TestRunner::instance().run_all();
}
