#include "test_framework.hpp"
#include "types.hpp"
#include "block.hpp"
#include "segment.hpp"
#include "mapping.hpp"
#include "metrics.hpp"
#include "lfs_simulator.hpp"
#include "synthetic_workload.hpp"

using namespace smartgc;

// -----------------------------------------------------------------------------
// Test 1: Block State Transitions
// -----------------------------------------------------------------------------
TEST_CASE(test_block_state_transitions) {
    Block b;
    ASSERT_TRUE(b.is_free());
    ASSERT_FALSE(b.is_valid());
    ASSERT_FALSE(b.is_invalid());
    ASSERT_EQ(b.lba, INVALID_LBA);

    // Allocate block
    b.allocate(100);
    ASSERT_FALSE(b.is_free());
    ASSERT_TRUE(b.is_valid());
    ASSERT_FALSE(b.is_invalid());
    ASSERT_EQ(b.lba, 100);

    // Invalidate block
    b.invalidate();
    ASSERT_FALSE(b.is_free());
    ASSERT_FALSE(b.is_valid());
    ASSERT_TRUE(b.is_invalid());
    ASSERT_EQ(b.lba, 100);

    // Reset block
    b.reset();
    ASSERT_TRUE(b.is_free());
    ASSERT_FALSE(b.is_valid());
    ASSERT_FALSE(b.is_invalid());
    ASSERT_EQ(b.lba, INVALID_LBA);
}

// -----------------------------------------------------------------------------
// Test 2: Segment Capacity, Allocations, Invalidation, and Reset
// -----------------------------------------------------------------------------
TEST_CASE(test_segment_operations) {
    Segment seg(0, 4);
    ASSERT_EQ(seg.id(), 0);
    ASSERT_EQ(seg.capacity(), 4);
    ASSERT_EQ(seg.valid_count(), 0);
    ASSERT_EQ(seg.invalid_count(), 0);
    ASSERT_EQ(seg.free_count(), 4);
    ASSERT_TRUE(seg.has_free_block());
    ASSERT_FALSE(seg.is_full());

    // Allocate 4 blocks
    size_t idx0 = seg.allocate_block(10);
    size_t idx1 = seg.allocate_block(20);
    size_t idx2 = seg.allocate_block(30);
    size_t idx3 = seg.allocate_block(40);

    ASSERT_EQ(idx0, 0);
    ASSERT_EQ(idx1, 1);
    ASSERT_EQ(idx2, 2);
    ASSERT_EQ(idx3, 3);
    ASSERT_EQ(seg.valid_count(), 4);
    ASSERT_EQ(seg.free_count(), 0);
    ASSERT_TRUE(seg.is_full());
    ASSERT_FALSE(seg.has_free_block());

    // Invalidate block 1 (LBA 20)
    seg.invalidate_block(1);
    ASSERT_EQ(seg.valid_count(), 3);
    ASSERT_EQ(seg.invalid_count(), 1);
    ASSERT_NEAR(seg.valid_ratio(), 0.75, 1e-6);
    ASSERT_NEAR(seg.invalid_ratio(), 0.25, 1e-6);

    // Reset segment
    seg.reset();
    ASSERT_EQ(seg.valid_count(), 0);
    ASSERT_EQ(seg.invalid_count(), 0);
    ASSERT_EQ(seg.free_count(), 4);
    ASSERT_FALSE(seg.is_full());
    ASSERT_TRUE(seg.has_free_block());
}

// -----------------------------------------------------------------------------
// Test 3: LBA Remapping and Out-of-Place Overwrite Correctness
// -----------------------------------------------------------------------------
TEST_CASE(test_lba_mapping_and_overwrites) {
    SimulatorConfig config;
    config.total_segments = 4;
    config.blocks_per_segment = 4;
    config.gc_free_segments_threshold = 1;
    config.block_size_bytes = 4096;

    LfsSimulator sim(config);

    // Initial write of LBA 5
    sim.write(5);
    ASSERT_TRUE(sim.read(5));
    ASSERT_FALSE(sim.read(99));

    PhysicalAddress addr1;
    ASSERT_TRUE(sim.mapping().get(5, addr1));
    ASSERT_EQ(addr1.segment_id, 0);
    ASSERT_EQ(addr1.block_index, 0);
    ASSERT_TRUE(sim.segments()[0].get_block(0).is_valid());

    // Overwrite LBA 5
    sim.write(5);
    PhysicalAddress addr2;
    ASSERT_TRUE(sim.mapping().get(5, addr2));
    ASSERT_EQ(addr2.segment_id, 0);
    ASSERT_EQ(addr2.block_index, 1);

    // Old location (0, 0) must be INVALID, new location (0, 1) must be VALID
    ASSERT_TRUE(sim.segments()[0].get_block(0).is_invalid());
    ASSERT_TRUE(sim.segments()[0].get_block(1).is_valid());

    // Metrics check before any GC has occurred
    const auto& m = sim.metrics();
    ASSERT_EQ(m.total_write_requests, 2);
    ASSERT_EQ(m.logical_bytes_written, 2 * 4096);
    ASSERT_EQ(m.physical_bytes_written, 2 * 4096);
    ASSERT_EQ(m.gc_bytes_copied, 0);
    ASSERT_EQ(m.gc_count, 0);
    ASSERT_NEAR(m.waf(), 1.0, 1e-6);
}

// -----------------------------------------------------------------------------
// Test 4: Greedy GC Victim Selection (Fewest Valid Blocks)
// -----------------------------------------------------------------------------
TEST_CASE(test_greedy_gc_victim_selection) {
    SimulatorConfig config;
    config.total_segments = 6;
    config.blocks_per_segment = 4;
    config.gc_free_segments_threshold = 1;

    LfsSimulator sim(config);

    // Segment 0: write LBAs 0, 1, 2, 3 (4 valid)
    for (int i = 0; i < 4; ++i) sim.write(i);

    // Segment 1: write LBAs 4, 5, 6, 7 (4 valid)
    for (int i = 4; i < 8; ++i) sim.write(i);

    // Segment 2: write LBAs 8, 9, 10, 11 (4 valid)
    for (int i = 8; i < 12; ++i) sim.write(i);

    // Now invalidate 3 blocks in Segment 0 by overwriting LBAs 0, 1, 2
    // (these go into Segment 3)
    sim.write(0);
    sim.write(1);
    sim.write(2);

    // Invalidate only 1 block in Segment 1 by overwriting LBA 4
    sim.write(4); // Fills Segment 3

    // At this point:
    // Segment 0: 1 valid (LBA 3), 3 invalid (LBAs 0, 1, 2)
    // Segment 1: 3 valid (LBAs 5, 6, 7), 1 invalid (LBA 4)
    // Segment 2: 4 valid (LBAs 8, 9, 10, 11), 0 invalid
    // Segment 3: 4 valid (LBAs 0, 1, 2, 4) - closed
    ASSERT_EQ(sim.segments()[0].valid_count(), 1);
    ASSERT_EQ(sim.segments()[1].valid_count(), 3);
    ASSERT_EQ(sim.segments()[2].valid_count(), 4);

    // Trigger GC directly
    bool gc_ran = sim.run_greedy_gc();
    ASSERT_TRUE(gc_ran);

    // Greedy policy MUST have chosen Segment 0 because it had only 1 valid block.
    // Segment 0 is now cleaned and reset to FREE state
    ASSERT_EQ(sim.segments()[0].valid_count(), 0);
    ASSERT_EQ(sim.segments()[0].invalid_count(), 0);
    ASSERT_EQ(sim.segments()[0].free_count(), 4);

    // LBA 3 (the single surviving valid block) was migrated to the active open segment
    ASSERT_TRUE(sim.read(3));
    PhysicalAddress addr;
    ASSERT_TRUE(sim.mapping().get(3, addr));
    ASSERT_NE(addr.segment_id, 0); // Moved out of Segment 0
    ASSERT_EQ(sim.metrics().valid_blocks_migrated, 1);
    ASSERT_EQ(sim.metrics().gc_count, 1);
}

// -----------------------------------------------------------------------------
// Test 5: Hand-Computable Deterministic WAF Verification
// -----------------------------------------------------------------------------
TEST_CASE(test_hand_computed_waf) {
    // -------------------------------------------------------------------------
    // Hand Calculation Step-by-Step Breakdown:
    // -------------------------------------------------------------------------
    // Configuration:
    //   Total Segments: 4 (IDs: 0, 1, 2, 3)
    //   Blocks per Segment: 4
    //   Block Size: 4096 bytes
    //   GC Trigger: when free segments <= 1
    //
    // Initial State:
    //   Open segment: 0
    //   Free pool: [1, 2, 3] (size = 3)
    //
    // Step 1: Write LBAs 0, 1, 2, 3 (4 writes)
    //   - Fills Segment 0 with LBAs 0..3 (4 valid)
    //   - Logical writes: 4 * 4KB = 16KB
    //   - Physical writes: 4 * 4KB = 16KB
    //
    // Step 2: Write LBAs 4, 5, 6, 7 (4 writes)
    //   - Open segment advances to Segment 1 (free pool: [2, 3], size = 2)
    //   - Fills Segment 1 with LBAs 4..7 (4 valid)
    //   - Logical writes: 8 * 4KB = 32KB
    //   - Physical writes: 8 * 4KB = 32KB
    //
    // Step 3: Overwrite LBAs 0, 1, 2, 3 (4 writes)
    //   - Open segment advances to Segment 2 (free pool: [3], size = 1)
    //   - Fills Segment 2 with new copies of LBAs 0..3 (4 valid)
    //   - In Segment 0: all 4 blocks become INVALID (valid = 0, invalid = 4)
    //   - Logical writes: 12 * 4KB = 48KB
    //   - Physical writes: 12 * 4KB = 48KB
    //
    // Step 4: Write LBA 8 (1 write)
    //   - Open segment 2 was full. Next segment from free pool is Segment 3.
    //   - Free pool size before popping is 1 <= threshold 1 -> GC triggers!
    //   - GC examines candidate closed segments:
    //       Segment 0: valid = 0, invalid = 4
    //       Segment 1: valid = 4, invalid = 0
    //   - Greedy chooses Segment 0 (0 valid blocks).
    //   - Migrated valid blocks: 0.
    //   - Segment 0 is erased and returned to free pool: [3, 0].
    //   - Open segment 3 is opened (free pool: [0]).
    //   - LBA 8 is written into Segment 3, block 0.
    //   - Logical writes: 13 * 4KB = 53,248 bytes
    //   - Physical writes: 13 * 4KB = 53,248 bytes (0 GC migrations so far)
    //   - GC Count: 1
    //
    // Step 5: Overwrite LBAs 4, 5, 6 (3 writes)
    //   - Written into Segment 3 (blocks 1, 2, 3). Segment 3 is now full!
    //   - In Segment 1: LBAs 4, 5, 6 become INVALID. LBA 7 remains VALID.
    //   - Segment 1 status: 1 valid (LBA 7), 3 invalid (LBAs 4, 5, 6).
    //   - Logical writes: 16 * 4KB = 65,536 bytes
    //   - Physical writes: 16 * 4KB = 65,536 bytes
    //
    // Step 6: Write LBA 9 (1 write)
    //   - Open segment 3 is full. Free pool has [0] (size = 1 <= threshold 1).
    //   - GC triggers!
    //   - GC examines candidate closed segments:
    //       Segment 1: valid = 1 (LBA 7), invalid = 3
    //       Segment 2: valid = 4 (LBAs 0, 1, 2, 3), invalid = 0
    //   - Greedy chooses Segment 1 (1 valid block).
    //   - GC opens Segment 0 from free pool to migrate valid blocks.
    //   - Migrates LBA 7 to Segment 0, block 0.
    //       -> valid_blocks_migrated = 1 (4,096 bytes)
    //       -> physical_bytes += 4096
    //       -> logical_bytes is NOT changed!
    //   - Segment 1 is erased and returned to free pool: [1].
    //   - LBA 9 is written into Segment 0, block 1.
    //       -> logical_bytes += 4096
    //       -> physical_bytes += 4096
    //
    // Final Totals:
    //   - Total user write requests: 17
    //   - Logical Bytes Written: 17 * 4096 = 69,632 bytes
    //   - Valid Blocks Migrated: 1 block = 4,096 bytes
    //   - Physical Bytes Written: (17 + 1) * 4096 = 18 * 4096 = 73,728 bytes
    //   - GC Invocations: 2
    //   - Theoretical Exact WAF: 18 / 17 = 1.0588235294117647...
    // -------------------------------------------------------------------------

    SimulatorConfig config;
    config.total_segments = 4;
    config.blocks_per_segment = 4;
    config.block_size_bytes = 4096;
    config.gc_free_segments_threshold = 1;

    LfsSimulator sim(config);

    // Step 1: Write LBAs 0..3
    for (int i = 0; i < 4; ++i) sim.write(i);

    // Step 2: Write LBAs 4..7
    for (int i = 4; i < 8; ++i) sim.write(i);

    // Step 3: Overwrite LBAs 0..3
    for (int i = 0; i < 4; ++i) sim.write(i);

    // Step 4: Write LBA 8 (triggers GC on Segment 0, 0 valid migrated)
    sim.write(8);

    // Step 5: Overwrite LBAs 4..6
    sim.write(4);
    sim.write(5);
    sim.write(6);

    // Step 6: Write LBA 9 (triggers GC on Segment 1, 1 valid migrated)
    sim.write(9);

    const auto& m = sim.metrics();

    // Verify exact counts match hand calculation
    ASSERT_EQ(m.total_write_requests, 17);
    ASSERT_EQ(m.logical_bytes_written, 17 * 4096);
    ASSERT_EQ(m.physical_bytes_written, 18 * 4096);
    ASSERT_EQ(m.gc_bytes_copied, 1 * 4096);
    ASSERT_EQ(m.valid_blocks_migrated, 1);
    ASSERT_EQ(m.gc_count, 2);

    double expected_waf = 18.0 / 17.0;
    ASSERT_NEAR(m.waf(), expected_waf, 1e-6);

    // Verify data integrity: all active LBAs must be readable and point to correct locations
    ASSERT_TRUE(sim.read(0));
    ASSERT_TRUE(sim.read(1));
    ASSERT_TRUE(sim.read(2));
    ASSERT_TRUE(sim.read(3));
    ASSERT_TRUE(sim.read(4));
    ASSERT_TRUE(sim.read(5));
    ASSERT_TRUE(sim.read(6));
    ASSERT_TRUE(sim.read(7)); // Migrated LBA 7 must still be valid and mapped!
    ASSERT_TRUE(sim.read(8));
    ASSERT_TRUE(sim.read(9));
}

int main() {
    return smartgc::test::TestRunner::instance().run_all();
}
