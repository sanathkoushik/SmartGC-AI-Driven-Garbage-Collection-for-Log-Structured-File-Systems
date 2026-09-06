# SmartGC Progress & Milestones

## Phase Status Summary

| Phase | Description | Status | Commit / Milestone Notes |
| :--- | :--- | :--- | :--- |
| **Phase 0** | Project setup, directory structure, docs, config, CMake skeleton, Git init | **Completed** | Commit `ca7a91d`: Repo initialized, contracts documented in `architecture.md`, `config.yaml` established. |
| **Phase 1** | Baseline LFS Simulator + Greedy GC (C++17), unit tests, synthetic workload, WAF verification | **Completed** | Full C++17 LFS engine implemented, 5 unit tests passed (including hand-verified WAF), baseline CLI executed. |
| **Phase 2** | Dataset inspection & raw trace normalization (Python) | **Pending** | Awaiting Phase 1 review and user go-ahead. |
| **Phase 3** | Historical rewrite-interval sequence construction & split (Python) | **Pending** | Awaiting Phase 2. |
| **Phase 4** | PyTorch lightweight LSTM training, evaluation & thresholding | **Pending** | Awaiting Phase 3. |
| **Phase 5** | Rule-based heuristic & LSTM prediction integration into simulator | **Pending** | Awaiting Phase 4. |
| **Phase 6** | Controlled multi-policy benchmark execution & raw metrics collection | **Pending** | Awaiting Phase 5. |
| **Phase 7** | Visualization plots & comprehensive comparative analysis writeup | **Pending** | Awaiting Phase 6. |

---

## Phase 1 Verification Summary

### 1. Unit Test Suite (`simulator/build/simulator_tests.exe`)
- `test_block_state_transitions`: **PASS** (`FREE -> VALID -> INVALID -> FREE` transitions verified).
- `test_segment_operations`: **PASS** (Block allocation, count updates, invalidation, segment reset).
- `test_lba_mapping_and_overwrites`: **PASS** (Out-of-place writes, L2P table updates, block invalidation).
- `test_greedy_gc_victim_selection`: **PASS** (Greedy victim selection by minimum valid blocks).
- `test_hand_computed_waf`: **PASS** (Deterministic 4-segment x 4-block test with 17 user writes, 2 GC cycles, exactly 1 valid block migrated, exact WAF = $18/17 \approx 1.058824$).

### 2. Baseline Synthetic Workload Run (`smartgc_sim.exe`)
- **Geometry**: 32 Segments, 64 Blocks/Segment (8MB capacity, 2048 blocks), Block Size: 4096 bytes.
- **Workload**: 5,000 requests, 500 LBAs, 80/20 Hot/Cold traffic skew.
- **Results**:
  - Logical Bytes Written: 20,480,000 bytes (19.53 MB)
  - Physical Bytes Written: 21,594,112 bytes (20.59 MB)
  - GC Migrated Bytes: 1,114,112 bytes (1.06 MB)
  - Valid Blocks Migrated: 272 blocks
  - GC Invocations: 53
  - **Baseline WAF (Mixed Placement)**: **1.0544**

---

## Decisions Log

1. **Decoupled Architecture**: C++ and Python subsystems interact exclusively through CSV contract files to eliminate binding dependencies and runtime complexity.
2. **Fixed Random Seeds**: All random generators (workloads, ML data split, model init) enforce explicit, logged seeds for bitwise reproducibility.
3. **Strict Novelty Boundary**: Hot/cold separation is documented as an established file system concept; SmartGC's scope is strictly evaluating sequence-based deep learning interval prediction vs. baseline heuristics in a controlled simulator.
4. **WAF Definition**: $\text{WAF} = \frac{\text{Physical Bytes Written}}{\text{Logical Bytes Written}}$. GC migration writes contribute solely to physical writes.
