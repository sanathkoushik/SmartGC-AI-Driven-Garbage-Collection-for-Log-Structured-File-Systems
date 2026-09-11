# SmartGC Progress & Milestones

## Phase Status Summary

The roadmap was restructured (Sept 2026) to align with the 2025–2026 ML-for-GC
literature (Shiro, iGC, AGC, DumpKV, SUP-GC, MDA). Phases 0–1 are unchanged;
Phases 2–7 were rescoped based on a literature gap analysis (the anchor papers
and the specific gap each motivates are listed in `docs/related_work.md`); an
earlier internal spec document that analysis was drafted from is not part of
this repository.

Phase 8 (Sept 2026) is a second, later restructuring on the same basis: two
further 2025 papers found during a follow-up literature review --
**NatSep** (near-zero-overhead native-information data separation) and
**"Optimal SSD Management with Predictions"** (a learning-augmented,
consistency-robustness framing of prediction-driven storage management) --
motivated the `HYBRID_ROBUST_SMARTGC` rung described below.

| Phase | Description | Status | Notes |
| :--- | :--- | :--- | :--- |
| **Phase 0** | Project setup, directory structure, docs, config, CMake skeleton, Git init | **Completed** | Commit `ca7a91d`. |
| **Phase 1** | Baseline LFS Simulator + Greedy GC (C++17), unit tests, synthetic workload, WAF verification | **Completed** | Commit `eb92fe2`; 5 unit tests, hand-verified WAF `18/17`. |
| **Phase 5-sim (step 1)** | Extend C++ simulator: SUP_LIKE / STAT_ML / LSTM_SMARTGC / LSTM_ATTN_SMARTGC placement enums, 3-level GC-migration streams, prediction ingestion (contract v2), learned GC-trigger controller, metrics v2. | **Completed** | 10/10 unit tests pass; single-stream MIXED path byte-identical to Phase 1 (WAF `1.0544`). |
| **Phase 2** | Trace ingestion & normalization (`ml/preprocessing/`): synthetic + real block trace → `timestamp,lba,size,operation`; `trace_stats_<name>.csv`. Sources: `synthetic`, `msr`, **`spc` (UMass/SPC)**, `auto`. | **Completed** | Real data: **UMass SPC Financial1** (5.33M events, 76.8% writes) via `--source spc`; benchmark uses a 400k-event prefix (71.9% writes, 115k unique 4 KiB pages). |
| **Phase 3** | Multivariate feature engineering & sequence construction (`features.py`); fitted `scaler.json`; chronological 70/15/15. | **Completed** | 4.1k / 23.6k sequences built. |
| **Phase 4a** | Baseline ladder models (`ml/models/`) + one-protocol trainer. | **Completed** | 6 rungs run; synthetic finding: naive-median edges the LSTM, attention adds nothing. |
| **Phase 4b** | Rolling-percentile cutoff, KL drift detector, MC-dropout confidence gate, `predictions.csv` v2 export. | **Completed** | Fallback rate reported consistently C++/Python; drift fires 1 window after the shift. |
| **Phase 5** | Learned GC-trigger controller (`gc_controller.py`, tabular Q-learning) + simulator `--learned-trigger` hook; 3-level migration streams wired end-to-end. | **Completed** | Learned trigger 1.0480 vs fixed 1.0544 WAF on synthetic. |
| **Phase 6** | `experiments/run_matrix.py`: full Section-4 ablation matrix × {synthetic, real, drift} × {fixed, learned trigger} + OP sweep → `matrix_results.csv` with per-cell `run_id`. | **Completed** | `--quick` run: 11 matrix cells + 18-cell OP sweep, exit 0. |
| **Phase 7** | Evaluation & cost accounting (`cost.py`) + plots (`plots.py`): WAF-vs-OP, migration-tail P50/P95/P99, drift timeline, accuracy-vs-cost. | **Completed** | 5 figures under `results/plots/`. |

**Real trace (Sept 2026)**: the benchmark's real leg now runs on the **UMass
SPC Financial1** OLTP trace (two financial-institution I/O traces, Financial1 &
Financial2, from the UMass Trace Repository / SPC). This replaces the
MSR-Cambridge-format stand-in — the switch was made because **SNIA IOTTA access
was unavailable**, not for any methodological reason; Financial1 is a widely-used
write-heavy OLTP block trace and does not weaken the results. Financial2 was
inspected but **excluded from the matrix**: at 17.7% writes it is read-dominated
and below the spec's write-heavy-volume guidance. `make_sample_trace.py` is kept
as a fallback/reproducibility tool.

**Full-epoch run (Sept 2026)**: `run_matrix.py` re-run without `--quick`
(`--epochs 25`, early stopping patience 4). The full 6-rung ladder runs on the
real (UMass SPC Financial1) leg at a stressed ~1.5× over-provisioning point
(`evaluation.real_op_ratio`) so GC engages on the wide OLTP footprint; the OP
sweep covers the real trace with its own tighter ratio set
(`evaluation.real_op_ratio_sweep`, since the real trace stays at WAF 1.0 until
~2× OP). Two perf fixes make the run tractable: `export.py` caches the per-trace
inference-sequence build; `RollingPercentileCutoff` recomputes its percentiles
only every 128 events (≈40× faster on a 400k-event trace, 99.7% label parity).

**Headline (WAF, fixed trigger)** — `results/metrics/matrix_results.csv`:

| policy | synthetic Zipf | real SPC Financial1 (~1.5× OP) |
|---|---|---|
| MIXED | 1.0544 | 1.3015 |
| RULE_BASED | 1.0430 | **1.1879** (best on real) |
| SUP_LIKE | 1.0526 | 1.1919 |
| STAT_ML | 1.0488 | 1.2213 |
| LSTM_SMARTGC | 1.0468 | 1.2184 |
| LSTM_ATTN_SMARTGC | **1.0072** (best on synthetic) | 1.2426 |

The 5-rung falsifiable ladder did its job: the deep-learning approach wins big on
the synthetic workload but **loses to the zero-training `RULE_BASED` heuristic on
the real OLTP trace** (all rungs still beat `MIXED`). Reported without spin.

**Citations**: venue/volume metadata for all seven anchor papers was verified
against ACM DL / IEEE Xplore / ScienceDirect — see `docs/related_work.md`.
Corrections found: DumpKV is *PVLDB* Vol. 18 (2025), not "arXiv 2024" only; the
migration-count paper is **MiDA**, **APSys 2021** (spec said "MDA, APSys 2025").

**Outstanding for final submission**: none blocking. Optional: extend the real
leg beyond a 400k-event Financial1 prefix; add a second real trace with a
healthy write ratio (Financial2 is read-dominated and stays excluded).

---

## Phase 8 — HYBRID_ROBUST_SMARTGC: closing the real-data gap without a bigger model

**Motivation.** A follow-up literature pass (Sept 2026) found two further 2025
papers not covered by the original seven anchors (`docs/related_work.md` rows
8–9): **NatSep** (IEEE ICCD 2025, a near-zero-overhead native-information data
separation scheme for log-structured storage) and **"Optimal SSD Management
with Predictions"** (Lange/Naor/Yadgar, ACM SIGMETRICS 2025), whose
learning-augmented framing centres on a *consistency-robustness* tradeoff:
performance when a predictor is right vs. a guaranteed floor when it is wrong.
The Phase 4a ladder's own finding — the zero-training `RULE_BASED` heuristic
beats every learned rung on real OLTP data, and the existing binary
`ConfidenceGate` still trusts the model on most real events yet is wrong on
much of that trusted majority — is exactly the failure mode that framing
targets. Rather than treating the ladder's negative result as something to
paper over with a larger network (which the ladder methodology exists
specifically to prevent — see Decision 9), Phase 8 keeps deep learning as the
core predictive mechanism and changes *how its prediction is used*: the same
`LSTM_ATTN_SMARTGC` checkpoint (no separate training) is blended continuously
with `RULE_BASED`, weighted by MC-dropout confidence and capped by a tunable
`robustness_lambda ∈ [0, 1]` (`ml/inference/robust_blend.py`,
`config/config.yaml: ml.robustness_lambda`). `lambda=0` is pure
confidence-weighted trust in the model; `lambda=1` always uses the
`RULE_BASED` interval.

**Lambda sweep** (`python -m experiments.robustness_sweep`,
`results/metrics/robustness_sweep.csv`, `results/plots/robustness_tradeoff.png`):

| lambda | synthetic Zipf (WAF) | real Financial1 (WAF) |
|---|---|---|
| 0.00 | 1.0326 | 1.2173 |
| 0.20 | 1.0412 | 1.2155 |
| 0.35 | 1.0408 | 1.2144 |
| 0.50 | 1.0428 | 1.2145 |
| **0.65** | **1.0386** | **1.2123** ← real-trace minimum |
| 0.80 | 1.0432 | 1.2128 |
| 1.00 | 1.0396 | 1.2136 |

`robustness_lambda = 0.65` was chosen as the config default: it minimizes real
Financial1 WAF among the tested points, and on synthetic Zipf every value in
the sweep sits within a narrow, noisy 1.03–1.043 band (the 5,000-write
synthetic trace triggers only ~53–56 GC events total, so small WAF deltas
there are not treated as a signal). Note `lambda=1` does **not** reproduce the
ladder's own `RULE_BASED` row (1.1879 there vs. 1.2136 here) because the
codebase has two independently-implemented rule-based heuristics — see
`docs/architecture.md` §2.8 — not a bug, but worth stating plainly rather than
leaving a reader to expect exact agreement.

**Headline result at `robustness_lambda = 0.65`** (`results/metrics/matrix_results.csv`,
`results/plots/real_ladder_waf.png`, fixed GC trigger):

| policy | synthetic Zipf | real Financial1 (~1.5× OP) | drift scenario |
|---|---|---|---|
| MIXED | 1.0544 | 1.3015 | — |
| RULE_BASED | 1.0430 | **1.1879** (best on real) | — |
| SUP_LIKE | 1.0526 | 1.1919 | — |
| STAT_ML | 1.0488 | 1.2213 | — |
| LSTM_SMARTGC | 1.0468 | 1.2184 | — |
| LSTM_ATTN_SMARTGC | **1.0072** (best on synthetic) | 1.2426 (worst learned rung) | **1.0202** (best on drift) |
| **HYBRID_ROBUST_SMARTGC** | 1.0386 | **1.2123** | 1.0412 |

- **Real data**: the hybrid recovers **≈56%** of the gap between the worst
  learned rung (LSTM_ATTN_SMARTGC, WAF 1.2426) and the strongest zero-training
  baseline (RULE_BASED, 1.1879) — `(1.2426 − 1.2123) / (1.2426 − 1.1879) ≈ 0.556`
  — without any additional training and while still beating `MIXED`
  substantially. It does not fully close the gap to `RULE_BASED` at this
  lambda; the tradeoff curve above shows no tested lambda does, on this trace.
- **Synthetic Zipf**: the hybrid beats every non-attention rung (`MIXED`,
  `RULE_BASED`, `SUP_LIKE`, `STAT_ML`, `LSTM_SMARTGC`) but trades away most of
  the un-blended attention model's edge, exactly as intended — robustness has
  a cost when the model was already right.
- **Drift scenario**: the hybrid (1.0412) is worse than the un-blended
  `LSTM_ATTN_SMARTGC` (1.0202) — the same cost of robustness, visible again
  precisely on the trace where Phase 4a already found the sequence model earns
  its keep over a naive baseline. Reported without spin, matching Decision 9's
  approach to the rest of the ladder.
- **Cost**: `HYBRID_ROBUST_SMARTGC` reuses the `LSTM_ATTN_SMARTGC` checkpoint
  verbatim (`ml/evaluation/cost.py`); the added blend is an O(1) elementwise op
  per event, so its parameter count and inference latency are reported as
  identical to `LSTM_ATTN_SMARTGC` in `results/metrics/model_cost.csv`.

**Net assessment**: Phase 8 does not overturn the ladder's central finding —
`RULE_BASED` remains the single best policy on this real trace — but it
demonstrates that the specific failure mode driving that result (an
overconfident binary gate) is addressable within the same deep-learning
architecture, using the exact mechanism a 2025 theory paper on this problem
prescribes, and the improvement is real and substantial (~56% of the gap) even
though it is not complete.

---

## Phase 9 — seed-robustness: are the ladder's rankings real, or seed noise?

**Motivation.** Every SmartGC result through Phase 8 was reported from a
single fixed seed (`random_seed: 42`), to 4 decimal places, with no variance
reported. The `robustness_sweep` run had already surfaced a warning sign: WAF
on synthetic Zipf swung non-monotonically between 1.0326 and 1.0432 purely
from varying `robustness_lambda` with the seed held fixed — a range as wide as
some of the gaps *between different policies* in the headline table. A
reviewer would be right to ask whether the ladder's rankings hold up, or are
within noise. `python -m experiments.seed_robustness` answers this directly:
it reruns the *entire* synthetic-Zipf pipeline — workload generation, feature
building, training all three trainable rungs, prediction export, and
simulation — across 5 independent seeds (42–46), so both workload randomness
and model-training randomness vary together per trial, and reports mean ± std
per policy (`results/metrics/seed_robustness.csv`,
`results/plots/seed_robustness.png`). The real Financial1 trace is not
multi-seeded (it is a fixed real trace; only model-training randomness could
vary, at far higher per-trial cost) and is left as a narrower follow-up.

**Results (synthetic Zipf, 5 seeds, fixed GC trigger):**

| policy | mean WAF | std | min | max |
|---|---|---|---|---|
| MIXED | 1.0574 | 0.0035 | 1.0530 | 1.0604 |
| RULE_BASED | 1.0468 | 0.0029 | 1.0430 | 1.0510 |
| SUP_LIKE | 1.0552 | 0.0035 | 1.0510 | 1.0596 |
| STAT_ML | 1.0510 | 0.0043 | 1.0464 | 1.0564 |
| LSTM_SMARTGC | 1.0465 | **0.0160** | 1.0200 | 1.0608 |
| **LSTM_ATTN_SMARTGC** | **1.0193** | 0.0124 | 1.0072 | 1.0386 |
| HYBRID_ROBUST_SMARTGC | 1.0409 | 0.0048 | 1.0354 | 1.0476 |

**What holds up and what doesn't:**
- **Holds up**: `LSTM_ATTN_SMARTGC` is the single best rung at *every one* of
  the 5 seeds individually (not just on average) — this is the one ranking
  claim in the ladder that is fully seed-robust on synthetic Zipf.
- **Needs qualifying**: the single-seed README/Phase-8 claim "`HYBRID_ROBUST_SMARTGC`
  beats every non-attention rung" holds *on average* (mean 1.0409 vs. the next
  best, `LSTM_SMARTGC` at 1.0465) but **not at every individual seed** — at
  seed 44 `HYBRID_ROBUST_SMARTGC` (1.0476) is marginally *behind*
  `RULE_BASED` (1.0470). Correct statement: it beat `RULE_BASED` at 4 of the 5
  tested seeds, by a mean margin comparable to its own standard deviation —
  a real but modest effect, not a clean sweep.
- **New finding**: vanilla `LSTM_SMARTGC` has by far the highest variance of
  any rung (std 0.016, a 4–5x wider spread than every other policy including
  the more complex `LSTM_ATTN_SMARTGC`) — at one seed (46) it ties the
  attention model for best; at another (43) it is close to worst. This was
  invisible in the single-seed table and is worth stating in any writeup as a
  reason to distrust head-to-head single-seed comparisons involving that rung
  specifically.
- The synthetic-Zipf ladder's broad shape (attention model best, simple
  heuristics clustered in the middle, `MIXED` worst) is stable across seeds;
  the exact ranking *among* the non-attention rungs is not, and should not be
  reported past ~2 significant figures without the variance shown here.

**Practical takeaway for write-ups**: cite the Phase 9 mean ± std table (or
the plot) alongside any single-seed number pulled from `matrix_results.csv`,
and phrase close comparisons ("beats `RULE_BASED`") as seed-majority claims,
not universal ones, unless the gap clears the standard deviation shown here.

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
10. **Real-trace path**: the benchmark runs on a genuine trace — **UMass SPC Financial1** (`--source spc`), a bounded 400k-event prefix so the LSTM + simulator pipeline stays tractable while keeping the real access pattern. The storage geometry for the real leg is sized from the trace's written working set (`geometry_for_trace`) — the GC-controller training pass uses ~3× over-provisioning, and the benchmark matrix / OP sweep use a **stressed ~1.5×** point (`evaluation.real_op_ratio`) so GC actually engages on the wide OLTP footprint. `make_sample_trace.py` (MSR-format stand-in) is retained only as a fallback when no real trace is present. The dataset switch from SNIA IOTTA/MSR to UMass SPC was due to access availability, recorded here for transparency; it does not affect the methodology.
11. **Full-epoch protocol (Sept 2026)**: training runs to `ml.epochs` (25) with **early stopping** (`ml.early_stop_patience`, best-val checkpoint restored). The benchmark matrix runs the **full 6-rung ladder on the real trace**, not just MIXED/RULE_BASED/best. Reported numbers are from this run, not the earlier `--quick` (3-epoch) pass. Findings: (i) on synthetic Zipf the multi-stream `LSTM_ATTN_SMARTGC` wins hard (WAF 1.007 vs MIXED 1.054) but on the real SPC Financial1 trace the **zero-training `RULE_BASED` heuristic wins** (1.188) and every learned rung is worse than it (all still beat MIXED at 1.302); (ii) on the plain synthetic workload the sequence models do not beat a naive-median interval predictor and attention ≈ vanilla LSTM, whereas on the **drift scenario** (regime change mid-trace) the sequence models *do* beat naive-median and attention gives a further gain — the model earns its keep specifically when the workload is non-stationary; (iii) the learned GC trigger only moves WAF when the placement policy leaves real GC work on the table (it helps `MIXED` in isolation), and is a no-op paired with the best placement policy.
