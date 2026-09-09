# SmartGC Progress & Milestones

## Phase Status Summary

| Phase | Description | Status | Notes |
| :--- | :--- | :--- | :--- |
| **Phase 0** | Project setup, directory structure, docs, config, CMake skeleton, Git init | **Completed** | Commit `ca7a91d` |
| **Phase 1** | Baseline LFS Simulator + Greedy GC (C++17), unit tests, synthetic workload | **Completed** | Commit `eb92fe2` |
| **Phase 1b** | Foundation audit and correctness fixes (GC reserve, admission control, config loading, contracts) | **Completed** | This entry |
| **Phase 2** | Dataset acquisition, provenance and raw trace normalization (Python) | **In progress** | Real MSR/FIU traces; sample-trace fixture available |
| **Phase 3** | Rewrite-interval sequence construction & chronological split | **Pending** | |
| **Phase 4** | LSTM pretraining, fine-tuning, evaluation & thresholding | **Pending** | |
| **Phase 5** | Rule-based heuristic & LSTM prediction integration into the simulator | **Pending** | |
| **Phase 6** | Controlled multi-policy benchmark execution | **Pending** | |
| **Phase 7** | Plots, tables and comparative analysis | **Pending** | |

---

## Phase 1b — Foundation Audit and Fixes

The committed Phase-1 engine was re-validated before any later phase was built. It reproduced its documented baseline **exactly** (WAF 1.0544, 272 blocks migrated, 53 GC invocations, 20,480,000 logical / 21,594,112 physical bytes), and all 5 original unit tests passed. Five genuine defects were nevertheless found, all of which would have blocked or corrupted the experiments.

### Defects found and fixed

**1. Cleaning could exhaust the free pool at high utilization (blocking).**
`migrate_valid_block` drew fresh segments from the same pool as user writes, so at high utilization a cleaning pass could run out of space mid-migration. Measured behaviour of the old engine on a skewed workload:

| Live utilization | Old engine |
| :--- | :--- |
| 24.4% | WAF 1.0544 |
| 50.0% | WAF 1.5050 |
| 75.0% | WAF 2.5367 |
| 90.0% | WAF 6.6240 |
| **95.0%** | **aborted: "No free segment available to receive migrated valid block"** |
| 100.0% | aborted |

This mattered because hot/cold separation only has headroom at high utilization — precisely where the engine died.

*Fix*: `simulator.gc_reserved_segments` (>= 1), a pool of segments only the collector may consume, plus a dedicated `GC` append point. A victim is never fully valid, so it holds at most `blocks_per_segment - 1` survivors and one reserved segment always absorbs a pass. See `docs/architecture.md` §3.6.

**2. Fully valid segments could be selected as GC victims.**
Cleaning such a segment migrates a whole segment's worth of data and frees nothing. They are now excluded, which also gives the cleaning loop a termination proof: every pass strictly increases the number of free blocks.

**3. No admission control.**
Writing more distinct LBAs than the device could hold surfaced as an obscure failure inside the collector. A new LBA beyond `max_live_blocks` now raises `DeviceFullError` naming the live count and the limit.

**4. `write()` silently ignored the request size.**
Every request was accounted as one block regardless of its length, so an unexpanded 128 KB request would have been counted as 4 KB. `write()` now rejects any request that is not exactly one block; preprocessing is responsible for block expansion.

**5. `config.yaml` was never read, and the metrics row hardcoded `"MIXED"`.**
The README claimed "no hardcoded parameters" while the C++ used its own defaults. A strict loader for the config subset the simulator uses now exists (`simulator/src/config.cpp`); unknown keys inside a recognised section are a hard error rather than a silent fallback. The metrics row reports the actual placement policy and the full run descriptor.

### Defect found and fixed during this phase's own validation

**6. A shared GC stream stranded user segments.**
With `gc_separate_stream: false`, a cleaning pass migrates into the *user* append point. The user write path then opened a *second* segment, leaving the first flagged open forever — permanently excluded from victim selection. The device reported itself full after ~3,500 writes at 24% utilization. The write path now adopts a segment that cleaning already opened. Regression test: `test_shared_gc_stream_does_not_strand_segments`.

### Consequence for the Phase-1 baseline number

The GC append point changes where survivors land, so the headline baseline moves slightly. All three numbers below are measured on the identical workload (5,000 requests, 500 LBAs, seed 42, 32x64 geometry):

| Engine | Valid blocks migrated | GC invocations | WAF |
| :--- | ---: | ---: | ---: |
| Phase 1 as committed (`eb92fe2`) | 272 | 53 | 1.0544 |
| Phase 1b, `gc_separate_stream: false` | 266 | 54 | 1.0532 |
| Phase 1b, `gc_separate_stream: true` (new default) | 236 | 53 | **1.0472** |

The old and new shared-stream figures agree to within 0.1%; the residual difference is the GC reserve holding two segments back from user writes. Separating the GC stream reduces migrations by 13% on its own.

### Measured utilization sweep (Phase 1b engine)

50,000 requests, 32 segments x 64 blocks (2,048 blocks raw), MIXED placement, separate GC stream, seed 42. Recorded in `results/metrics/foundation_validation.csv`.

| Target utilization | Valid blocks migrated | GC invocations | WAF |
| ---: | ---: | ---: | ---: |
| 25.0% | 5,183 | 833 | 1.1037 |
| 50.0% | 19,836 | 1,062 | 1.3967 |
| 75.0% | 74,495 | 1,916 | 2.4899 |
| 85.0% | 169,592 | 3,402 | 4.3918 |
| 87.5% (usable maximum) | 229,371 | 4,336 | 5.5874 |

**Experimental consequence**: the previous default working set (500 LBAs = 24% utilization) leaves almost no migration cost for SmartGC to reduce. `config/config.yaml` now defaults to a 1,536-block working set (75% utilization, WAF 2.49), which is a meaningful operating point. The policy comparisons will be run across a range of utilizations.

### Test suite

`simulator/build/simulator_tests` — **18 passed, 0 failed**. The 5 original tests are retained (with the hand-computed WAF trace recomputed for the corrected engine: 21 user writes, 2 migrations, 2 GC passes, exact WAF = 23/21 = 1.095238). New coverage: fully-valid-victim exclusion, high-utilization completion, admission control, request-size rejection, GC stream separation, temperature-routes-placement-only, config validation, config.yaml loading including malformed input, GC-migrates-only-valid-blocks, run determinism, metrics CSV contract, and the shared-stream regression.

### Build and toolchain

- CMake 4.4.3 and Python 3.12.10 installed via `winget`; `simulator/CMakeLists.txt` remains authoritative.
- `scripts/build_simulator.py` locates CMake (including winget install paths not yet on PATH) and selects the MinGW Makefiles generator when the available compiler is MSYS2 GCC.
- `CMakeLists.txt` now links the GCC runtime statically on MinGW. This is not cosmetic: the reference machine's MSYS2 installation cannot link the *shared* runtime at all — `ld` exits with status 116 and no diagnostic for any translation unit that uses exceptions.
- Compiles clean under `-Wall -Wextra -Wpedantic`, zero warnings.

---

## Decisions Log

1. **Decoupled Architecture**: C++ and Python subsystems interact exclusively through CSV contract files.
2. **Fixed Random Seeds**: All random generators enforce explicit, logged seeds.
3. **Strict Novelty Boundary**: Hot/cold separation is an established file system concept. SmartGC's scope is evaluating whether a sequence model's rewrite-interval predictions beat simple heuristics, measured end-to-end in a controlled simulator.
4. **WAF Definition**: `physical / logical`. GC migrations contribute to physical writes only.
5. **Over-provisioning is explicit** (Phase 1b): the collector owns a reserve of segments, and usable capacity is derived from it rather than assumed. Placement policies that need two user append points have one segment less of usable capacity, so geometry and working set are fixed across every policy in a comparison.
6. **GC gets its own append point** (Phase 1b), so that data which survived cleaning is not re-interleaved into a user log. `gc_separate_stream: false` is retained as an ablation.
7. **Real data only for results** (Phase 1b): synthetic workloads are for unit tests and development. Every conference number comes from a real trace and is labelled with its dataset.
