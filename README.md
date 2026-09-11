# SmartGC: AI-Driven Garbage Collection for Log-Structured File Systems

## Project Overview

In a Log-Structured File System (LFS), data is written sequentially to log segments rather than updated in place. As files are modified and deleted over time, storage segments accumulate a mixture of obsolete (**invalid**) and active (**valid**) blocks. To reclaim storage space, a Garbage Collector (GC) must identify victim segments, copy their remaining valid blocks to the head of the log (valid-data migration), and erase the segment for reuse. 

When a segment co-locates **hot** data (frequently rewritten) and **cold** data (rarely rewritten), the hot blocks invalidate rapidly while cold blocks remain valid. When GC reclaims such mixed segments, it is forced to repeatedly copy the surviving cold data forward. This repeated copying of valid data directly drives **Write Amplification (WAF)**:

$$\text{WAF} = \frac{\text{Physical Bytes Written}}{\text{Logical Bytes Written}}$$

**SmartGC** investigates whether a sequence-based deep learning model (LSTM) trained on historical block-level rewrite intervals can accurately predict future rewrite intervals and classify logical block addresses (LBAs) into **HOT** or **COLD** streams prior to placement. By placing predicted hot and cold blocks into segregated segments, SmartGC aims to minimize valid-data migration during garbage collection.

---

## Honest Scope & Academic Novelty Framing

Hot/cold data separation itself is **not novel** — modern production file systems
(such as Linux F2FS) and extensive flash-translation-layer (FTL) literature
already separate data by temperature. SmartGC does **not** claim otherwise. The
contribution is specifically about *where and how* the learning is applied, and
about the **rigour of the baseline comparison**, evaluated on both synthetic and
real traces.

### What SmartGC actually evaluates

1. **Placement-time *and* GC-time learning.** A sequence model predicts a
   continuous rewrite interval that drives (a) initial stream placement and
   (b) *multi-level* (SHORT / MEDIUM / LONG) routing of valid blocks at
   GC-migration time — not just a one-shot HOT/COLD label at write time. A
   small tabular Q-learning controller additionally decides *when* to trigger
   GC from `[free_segment_ratio, recent_write_rate, recent_waf]`, replacing the
   static low-watermark. Both GC-time pieces are independently ablatable.
2. **Dynamic vs. static thresholds.** The HOT/COLD cutoff is a rolling
   percentile over a sliding window of recent predictions, not a hand-set
   constant; a KL-divergence drift detector flags when the workload has shifted;
   a confidence gate falls back to the rule-based decision per block when the
   model is unsure, and the fallback rate is reported.
3. **A five-rung baseline ladder**, so the deep-learning claim is *tested*, not
   assumed: `MIXED` → `RULE_BASED` (zero-training threshold) → `SUP_LIKE`
   (SUP-GC: default-hot-on-write, cold-on-GC-copy, near-zero computation) →
   `STAT_ML` (a lightweight gradient-boosted regressor on the same features) →
   `LSTM_SMARTGC` (vanilla 2-layer LSTM) → `LSTM_ATTN_SMARTGC` (LSTM + additive
   attention, multi-stream). If a cheap rung wins, that is the reported result.
4. **Real-trace validation.** The benchmark's real leg runs on the **UMass SPC
   Financial1** OLTP block trace (`--source spc`; 5.33M events, 76.8% writes) in
   addition to the synthetic Zipf workload and a concatenated workload-drift
   scenario. A bounded 400k-event prefix is used so the LSTM + simulator pipeline
   stays tractable. `--source msr` / `--source auto` and a format-accurate
   stand-in (`make_sample_trace.py`) remain available. Financial2 was inspected
   and excluded (17.7% writes — read-dominated).
5. **Accuracy-vs-inference-cost reporting.** Parameter count, inference latency,
   throughput and memory are reported next to the WAF numbers, so the
   host-side-ML-overhead question (raised by in-storage-inference work such as
   Shiro) can be answered honestly rather than ignored.
6. **Phase 8 — closing the real-data gap with a learning-augmented hybrid,
   not a bigger model.** The 6-rung ladder itself falsified "deep learning
   helps" on real OLTP data (below). Rather than treating that as something to
   paper over with a larger network, `HYBRID_ROBUST_SMARTGC` reuses the exact
   same LSTM+attention checkpoint but replaces the binary confidence-fallback
   gate with a continuous, confidence-weighted blend against `RULE_BASED`,
   capped by a tunable `robustness_lambda` — a direct, empirically-tuned
   instantiation of the *consistency-robustness* tradeoff from Lange, Naor &
   Yadgar's "Optimal SSD Management with Predictions" (ACM SIGMETRICS 2025;
   `docs/related_work.md`). Deep learning remains the core predictive
   mechanism throughout; Phase 8 changes how its prediction is *used* at
   inference time, not what predicts it.

### Honest findings (full-epoch run, `results/metrics/matrix_results.csv`)

**Write Amplification, fixed GC trigger:**

| Placement policy | synthetic Zipf | real — UMass SPC Financial1 (~1.5× OP) |
|---|---|---|
| `MIXED` (baseline) | 1.0544 | 1.3015 |
| `RULE_BASED` (zero-training) | 1.0430 | **1.1879** ← best on real |
| `SUP_LIKE` (SUP-GC, ~0 compute) | 1.0526 | 1.1919 |
| `STAT_ML` (sklearn GBDT) | 1.0488 | 1.2213 |
| `LSTM_SMARTGC` (vanilla LSTM) | 1.0468 | 1.2184 |
| `LSTM_ATTN_SMARTGC` (LSTM+attn, multi-stream) | **1.0072** ← best on synthetic | 1.2426 |
| `HYBRID_ROBUST_SMARTGC` (Phase 8, robustness-blend, λ=0.65) | 1.0386 | **1.2123** |

`HYBRID_ROBUST_SMARTGC` reuses the `LSTM_ATTN_SMARTGC` checkpoint verbatim (no
extra training) at `robustness_lambda=0.65`, chosen empirically as the
real-trace WAF minimum from a lambda sweep (`results/plots/robustness_tradeoff.png`).
On real data it recovers **≈56%** of the gap between the worst learned rung
(`LSTM_ATTN_SMARTGC`, 1.2426) and `RULE_BASED` (1.1879) while still beating
`MIXED` (1.3015) substantially — a genuine, measured improvement, though it
does not fully close the gap to `RULE_BASED`. On synthetic Zipf it beats every
non-attention rung *on average*; on the drift scenario (not shown above; see
`docs/progress.md` Phase 8) it costs a little relative to the un-blended
model — exactly the theoretically expected price of added robustness when the
model was already right.

**A note on single-seed numbers.** Every table above (and every number in
`matrix_results.csv`) is from one fixed seed (42). A dedicated 5-seed check
(Phase 9, `python -m experiments.seed_robustness`,
`results/plots/seed_robustness.png`) reruns the whole synthetic-Zipf pipeline
per seed and finds: `LSTM_ATTN_SMARTGC` best at every seed individually (a
robust claim); `HYBRID_ROBUST_SMARTGC` beats `RULE_BASED` at 4 of 5 seeds by a
margin comparable to its own standard deviation (a real but modest effect, not
a clean sweep); and vanilla `LSTM_SMARTGC` has ~4–5x the seed-to-seed variance
of every other rung, occasionally tying the attention model and occasionally
landing near the bottom. See `docs/progress.md` Phase 9 for the full mean±std
table before quoting any close single-seed comparison.

- **The ladder falsifies the "deep learning helps" hypothesis on real data.** On
  synthetic Zipf the multi-stream LSTM+attention is far ahead (WAF 1.007 vs
  1.054); on the real OLTP trace the **zero-training `RULE_BASED` heuristic wins**
  (1.188), and every learned rung is *worse* than it — though all still beat
  `MIXED` substantially. The synthetic win was partly an artifact of the
  generator's clean structure.
- On synthetic Zipf the per-LBA intervals are near-memoryless: a naive-median
  predictor edges the LSTM on interval MAE and **additive attention adds nothing**
  over the vanilla LSTM. On the **drift scenario** (regime change mid-trace) the
  sequence models *do* beat naive-median and attention gives a further gain — the
  model earns its keep specifically when the workload is non-stationary.
- On real data the **confidence gate fires ~31%** of the time (vs ~9.5% on
  synthetic) — the model knows it is uncertain — yet still underperforms pure
  `RULE_BASED`, i.e. it is also wrong on much of the 69% it trusts itself on.
- `SUP_LIKE` (near-zero computation) ≈ `RULE_BASED` on real OLTP (1.192 vs
  1.188), corroborating SUP-GC's "simplicity" thesis.
- The **learned GC trigger** helps only when the placement policy leaves real GC
  work on the table (e.g. it cut `MIXED` synthetic WAF ~1.054→1.048 in isolation);
  paired with the best placement policy — which already drives GC to near zero —
  it is a no-op. Kept strictly optional (`gc.learned_trigger`).
- **Cost** (`results/metrics/model_cost.csv`): `STAT_ML` ~9–12k params,
  `LSTM_SMARTGC` ~54k, `LSTM_ATTN_SMARTGC` ~56k; batch inference 2–9 ms/batch on
  CPU. On this evidence the LSTMs' inference cost is not bought back by a WAF win
  on real data.

### Reproducing the numbers

```
python -m experiments.run_matrix --op-sweep      # full Section-4 matrix + OP sweep
python -m experiments.robustness_sweep            # Phase 8: lambda tradeoff curve
python -m experiments.seed_robustness             # Phase 9: 5-seed mean+-std (synthetic Zipf)
python -m ml.evaluation.cost                      # accuracy-vs-cost table
python -m ml.evaluation.plots                     # comparative figures
```
Every metrics row carries a `run_id` = hash of that cell's effective config. To
add just the Phase 8 rung's rows to an existing `matrix_results.csv` without
re-running (or duplicating) the rest of the ladder:
```
python -m experiments.run_matrix --only-policy HYBRID_ROBUST_SMARTGC
```

---

## Core Concept: Hot/Cold vs. Valid/Invalid

An essential design principle in SmartGC is the independence of validity and temperature:
- **Valid vs. Invalid**: Represents current block state. A block is valid if it holds the most recent version of an LBA; it is invalid if it has been overwritten. Garbage collection decisions (which segments to clean and which blocks to migrate) are driven strictly by **validity**.
- **Hot vs. Cold**: Represents update frequency / temporal lifetime. Hot blocks are expected to be rewritten soon; cold blocks persist across long horizons. Hot/cold classification influences **placement** (which segment receives the write), isolating cold data from early-invalidating hot data to reduce future migrations.

---

## System Architecture & End-to-End Pipeline

The project is decoupled into two independent components communicating strictly via CSV files:

```
Raw I/O Trace  (synthetic Zipf  |  UMass SPC Financial1 real trace  |  drift scenario)
      │
      ▼
[Phase 2: ml/preprocessing/normalize.py]  ── trace_stats_<name>.csv
      │
      ▼ (normalized_trace.csv: timestamp, lba, size, operation   — size in 4 KiB blocks)
[Phase 3: ml/preprocessing/features.py]  ── ml/models/scaler.json (fitted once)
      │  multivariate per-LBA sequences, chronological 70/15/15
      ▼
[Phase 4a: baseline ladder — ml/models/, ml/training/train.py]
   RULE_BASED · SUP_LIKE · STAT_ML(sklearn) · LSTM_SMARTGC · LSTM_ATTN_SMARTGC(PyTorch)
      │
      ▼
[Phase 4b: ml/inference/export.py]  rolling percentile cutoff · drift detector · confidence gate
      │  Phase 8: HYBRID_ROBUST_SMARTGC — same checkpoint, robustness-blend inference (ml/inference/robust_blend.py)
      │
      ▼ (predictions.csv v2: …, predicted_stream_class, confidence)   ── drift_status_<name>.json
[Phase 5: C++17 LFS Simulator — simulator/]
      │  placement policies (incl. MIXED baseline) · 3-level GC-migration streams
      │  optional learned GC trigger  ◄── gc_policy_<name>.csv  (ml/training/gc_controller.py)
      ▼ (matrix_results.csv v2: waf, migration_stream_count, gc_migrated_p50/95/99, …)
[Phase 6: experiments/run_matrix.py]   full ablation matrix + over-provisioning sweep
      │
      ▼
[Phase 7: ml/evaluation/cost.py + plots.py]
      └── WAF · migration-tail P50/P95/P99 · WAF-vs-OP curves · accuracy-vs-inference-cost
```

---

## Repository Structure

```
SmartGC/
├── README.md
├── .gitignore
├── data/
│   ├── raw/                  # Raw block I/O trace files
│   ├── processed/            # Normalized traces (CSV contract)
│   └── predictions/          # Model and heuristic prediction outputs
├── simulator/
│   ├── CMakeLists.txt        # Simulator CMake build configuration
│   ├── include/              # Simulator headers (Block, Segment, Mapping, Metrics, LfsSimulator)
│   ├── src/                  # Simulator implementation & CLI runner
│   └── tests/                # Unit test suite and deterministic WAF verification
├── ml/
│   ├── common/               # config.yaml loader + CSV contract constants
│   ├── preprocessing/        # normalize.py, features.py, trace_stats.py, make_sample_trace.py
│   ├── models/               # baseline ladder: rule_based, sup_like, stat_ml, lstm, lstm_attention
│   ├── training/             # train.py, gc_controller.py (learned trigger), retrain_now.py
│   ├── inference/            # rolling_cutoff, drift, confidence_gate, robust_blend (Phase 8), export.py
│   └── evaluation/           # cost.py (accuracy-vs-cost), plots.py
├── experiments/              # run_matrix.py (Section-4 matrix), robustness_sweep.py (Phase 8),
│                             #   seed_robustness.py (Phase 9: multi-seed mean+-std)
├── results/
│   ├── metrics/              # CSV output metrics from simulator runs
│   └── plots/                # Comparative visualizations
├── scripts/                  # Helper automation scripts
├── tests/                    # pytest unit tests for ml/ (rolling cutoff, drift, confidence
│                             #   gate, robustness blend, LSTM+attention); run `pytest -q`
├── config/
│   └── config.yaml           # Centralized configuration (no hardcoded parameters)
└── docs/
    ├── architecture.md       # Detailed architectural specifications & CSV contracts
    └── progress.md           # Phase milestones and changelog
```

---

## Building and Running the C++ Simulator

### Prerequisites
- C++17 compliant compiler (`g++` 6.3+ or MSVC 2019+)
- CMake 3.15+ (or `python -m pip install cmake`)

### Build Instructions
```bash
# From workspace root
cmake -S simulator -B simulator/build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build simulator/build --config Release
```
The `CMAKE_EXPORT_COMPILE_COMMANDS` flag generates `simulator/build/compile_commands.json`,
which `.clangd` and `.vscode/c_cpp_properties.json` (both checked in) point editor
tooling (clangd, VS Code C/C++) at, so IntelliSense/diagnostics resolve the
project's headers and the compiler's standard library correctly.
`CMAKE_CXX_USE_RESPONSE_FILE_FOR_INCLUDES` is also forced `OFF` in
`simulator/CMakeLists.txt`: the MinGW Makefiles generator otherwise hides
every `-I` include path behind an `@....rsp` response file (to dodge Windows
command-line length limits), which clangd on Windows frequently fails to
resolve — showing up as false "cannot find `lfs_simulator.hpp`", "unknown
type `LbaType`", or "undeclared `std`" errors in the editor despite the real
build succeeding. Without a build directory yet, or after deleting one,
editors may show these same spurious errors — they are not compile errors;
run the two commands above and reload the editor (or restart its language
server) to clear them.

### Running Tests
```bash
./simulator/build/simulator_tests
```

### Running the Simulator CLI
```bash
./simulator/build/smartgc_sim --help

# Phase 1 baseline (single stream, fixed watermark):
./simulator/build/smartgc_sim --placement-policy MIXED --requests 5000 --lba-range 500

# Prediction-driven, 3-level GC migration, learned trigger:
./simulator/build/smartgc_sim --placement-policy LSTM_ATTN_SMARTGC \
    --trace data/processed/synthetic_zipf.csv \
    --predictions data/predictions/LSTM_ATTN_SMARTGC_synthetic_zipf.csv \
    --migration-streams 3 --confidence-threshold 0.55 \
    --learned-trigger --gc-policy-table data/predictions/gc_policy_synthetic_zipf.csv \
    --export-metrics results/metrics/run.csv
```

---

## Running the ML Pipeline (Phases 2–7)

### Prerequisites
Python 3.11+, and `pip install numpy pandas pyyaml scikit-learn torch matplotlib`.
All hyperparameters live in `config/config.yaml`; nothing is hard-coded in `ml/`.

### One-shot: the full benchmark matrix
```bash
python -m experiments.run_matrix --op-sweep      # add --quick for a fast dry run
python -m ml.evaluation.cost
python -m ml.evaluation.plots
```

### Step by step
```bash
# Phase 2 — normalize a trace + stats
python -m ml.preprocessing.make_sample_trace                       # MSR-format stand-in
python -m ml.preprocessing.normalize --source msr --input data/raw/msr_cambridge_src1.csv \
    --name msr_cambridge_src1 --dense-lba
python -m ml.preprocessing.trace_stats --input data/processed/msr_cambridge_src1.csv

# Phase 3 — features + sequences (writes ml/models/scaler.json)
python -m ml.preprocessing.features --input data/processed/synthetic_zipf.csv --name synthetic_zipf

# Phase 4a — train a rung
python -m ml.training.train --dataset synthetic_zipf --model LSTM_ATTN_SMARTGC

# Phase 4b — export predictions.csv (rolling cutoff + drift + confidence gate)
python -m ml.inference.export --trace data/processed/synthetic_zipf.csv \
    --model LSTM_ATTN_SMARTGC --streams 3

# Phase 5 — train the learned GC-trigger Q-table
python -m ml.training.gc_controller --trace data/processed/synthetic_zipf.csv --name synthetic_zipf
```
