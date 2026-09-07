# SmartGC Progress & Milestones

## Phase Status Summary

The roadmap was restructured (Sept 2026) to align with the 2025–2026 ML-for-GC
literature (Shiro, iGC, AGC, DumpKV, SUP-GC, MDA). Phases 0–1 are unchanged;
Phases 2–7 were rescoped. See `SmartGC_Implementation_Spec.md` for the gap
analysis this restructuring is based on.

| Phase | Description | Status | Notes |
| :--- | :--- | :--- | :--- |
| **Phase 0** | Project setup, directory structure, docs, config, CMake skeleton, Git init | **Completed** | Commit `ca7a91d`. |
| **Phase 1** | Baseline LFS Simulator + Greedy GC (C++17), unit tests, synthetic workload, WAF verification | **Completed** | Commit `eb92fe2`; 5 unit tests, hand-verified WAF `18/17`. |
| **Phase 5-sim (step 1)** | Extend C++ simulator: SUP_LIKE / STAT_ML / LSTM_SMARTGC / LSTM_ATTN_SMARTGC placement enums, 3-level GC-migration streams, prediction ingestion (contract v2), learned GC-trigger controller, metrics v2. | **Completed** | 10/10 unit tests pass; single-stream MIXED path byte-identical to Phase 1 (WAF `1.0544`). |
| **Phase 2** | Trace ingestion & normalization (`ml/preprocessing/`): synthetic + MSR/FIU real trace → `timestamp,lba,size,operation`; `trace_stats_<name>.csv`. | **Completed** | Verified on 5k synthetic Zipf + 40k MSR-format stand-in. |
| **Phase 3** | Multivariate feature engineering & sequence construction (`features.py`); fitted `scaler.json`; chronological 70/15/15. | **Completed** | 4.1k / 23.6k sequences built. |
| **Phase 4a** | Baseline ladder models (`ml/models/`) + one-protocol trainer. | **Completed** | 6 rungs run; synthetic finding: naive-median edges the LSTM, attention adds nothing. |
| **Phase 4b** | Rolling-percentile cutoff, KL drift detector, MC-dropout confidence gate, `predictions.csv` v2 export. | **Completed** | Fallback rate reported consistently C++/Python; drift fires 1 window after the shift. |
| **Phase 5** | Learned GC-trigger controller (`gc_controller.py`, tabular Q-learning) + simulator `--learned-trigger` hook; 3-level migration streams wired end-to-end. | **Completed** | Learned trigger 1.0480 vs fixed 1.0544 WAF on synthetic. |
| **Phase 6** | `experiments/run_matrix.py`: full Section-4 ablation matrix × {synthetic, real, drift} × {fixed, learned trigger} + OP sweep → `matrix_results.csv` with per-cell `run_id`. | **Completed** | `--quick` run: 11 matrix cells + 18-cell OP sweep, exit 0. |
| **Phase 7** | Evaluation & cost accounting (`cost.py`) + plots (`plots.py`): WAF-vs-OP, migration-tail P50/P95/P99, drift timeline, accuracy-vs-cost. | **Completed** | 5 figures under `results/plots/`. |

**Outstanding for final submission**: (a) full-epoch training run (the numbers
above are from the fast `--quick` path); (b) swap the MSR stand-in for a genuine
SNIA IOTTA trace; (c) re-verify venue metadata for the related-work citations.

---

## Phase 1 Verification Summary

### 1. Unit Test Suite (`simulator/build/simulator_tests.exe`) — 10/10 PASS
- `test_block_state_transitions`, `test_segment_operations`,
  `test_lba_mapping_and_overwrites`, `test_greedy_gc_victim_selection`,
  `test_hand_computed_waf` (Phase 1, exact WAF `18/17`).
- `test_placement_policy_and_streams` — enum round-trip + `effective_stream_count`.
- `test_sup_like_multistream_migration` — SUP-GC default-hot-on-write, GC survivor
  routed to the cold stream head.
- `test_prediction_table_parsing` — predictions.csv v2 + v1 fallback.
- `test_gc_trigger_controller` — fixed watermark vs learned/adaptive fallback.
- `test_multistream_end_to_end_invariants` — data-integrity + accounting
  invariants over a 2k-write multi-stream run; deterministic replay.

### 2. Baseline Synthetic Workload Run (`smartgc_sim.exe`, MIXED)
- 32 Segments × 64 Blocks/Segment, 5,000 requests, 500 LBAs, 80/20 skew.
- **Baseline WAF (Mixed Placement): 1.0544** — unchanged after the Phase 5-sim
  refactor.

---

## Decisions Log

1. **Decoupled Architecture**: C++ and Python subsystems interact exclusively through CSV contract files to eliminate binding dependencies and runtime complexity.
2. **Fixed Random Seeds**: All random generators (workloads, ML data split, model init, Q-learning) enforce explicit, logged seeds for bitwise reproducibility.
3. **Strict Novelty Boundary**: Hot/cold separation is documented as an established file system concept; SmartGC's scope is the *placement + GC-time* application of learning, adaptive thresholds, and the rigour of a 5-rung baseline comparison on synthetic **and** real traces.
4. **WAF Definition**: $\text{WAF} = \frac{\text{Physical Bytes Written}}{\text{Logical Bytes Written}}$. GC migration writes contribute solely to physical writes.
5. **Contract versioning (Sept 2026)**: the CSV schema is versioned (`CONTRACT_VERSION = 2`). Predictions gained `predicted_stream_class` + `confidence`; the metrics row gained `contract_version`, `run_id`, stream/learned-trigger/tail-latency counters. v1 predictions are still accepted (stream derived from class, confidence = 1.0).
6. **Multi-level GC-migration streams**: the simulator keeps up to 3 concurrent log heads (SHORT/MEDIUM/LONG) and re-applies lifetime separation at *GC-migration time* using the last prediction on record for each LBA, mirroring Shiro's multi-destination migration. `migration_stream_count` is config-driven; a `stream_count + 1` free-segment reserve plus a repeated-cleanup loop keeps multi-stream runs deadlock-free. The single-stream path is byte-identical to Phase 1.
7. **Learned GC trigger is optional and ablatable**: `gc.learned_trigger` defaults to `false` (Phase 1 fixed watermark). The learned controller is tabular Q-learning over a 3×3×3 discretised `[free_ratio, write_rate, waf]` state, trained in Python against a mirror env and exported as a Q-table CSV; unvisited states fall back to a deterministic adaptive rule in C++.
8. **Dynamic thresholds over constants**: `hot_percentile_cutoff` is retained only as a warm-up seed; at runtime the HOT/COLD cutoff and the N-way stream edges are rolling percentiles over `hot_cutoff_window_events`. Drift is flagged by symmetric KL between consecutive windows; a per-block confidence gate falls back to RULE_BASED and its fallback rate is a reported metric.
9. **Baseline ladder must be able to falsify the DL claim**: `SUP_LIKE` (near-zero computation) and `STAT_ML` (cheap learning) sit between `RULE_BASED` and the LSTMs specifically so a "deep learning doesn't help here" outcome is observable — and on the synthetic workload it partly is (naive-median MAE < LSTM; attention ≈ vanilla LSTM). Reported without spin.
10. **Real-trace path, stand-in data**: real MSR/FIU traces are not vendored (size + SNIA IOTTA terms). `make_sample_trace.py` emits a format-accurate stand-in so the pipeline is runnable; a genuine file of the same name in `data/raw/` is a drop-in replacement.
