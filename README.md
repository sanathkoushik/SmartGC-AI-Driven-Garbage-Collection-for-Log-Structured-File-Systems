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
4. **Real-trace validation.** The pipeline ingests MSR-Cambridge-format block
   traces (SNIA IOTTA) in addition to the synthetic Zipf workload and a
   concatenated workload-drift scenario. A genuine trace is not vendored (size +
   licence); a format-accurate stand-in ships so the pipeline runs end-to-end.
5. **Accuracy-vs-inference-cost reporting.** Parameter count, inference latency,
   throughput and memory are reported next to the WAF numbers, so the
   host-side-ML-overhead question (raised by in-storage-inference work such as
   Shiro) can be answered honestly rather than ignored.

### Honest findings so far

- On the **synthetic Zipf** workload the per-LBA rewrite intervals are close to
  memoryless, so a naive-median predictor edges the LSTM on raw interval MAE and
  **additive attention adds nothing** over the vanilla LSTM — a negative result,
  reported as-is.
- The WAF improvement that *does* show up for `LSTM_ATTN_SMARTGC` comes
  substantially from the **multi-stream GC-time routing**, not from placement-time
  prediction accuracy. The ladder is built so this distinction is visible; the
  2-stream rungs (`STAT_ML`, `LSTM_SMARTGC`) isolate the placement-only effect.
- The learned GC trigger gives a small, workload-dependent WAF change (helps on
  the plain synthetic workload, neutral-to-slightly-worse elsewhere in the quick
  runs); it is kept strictly optional.

### Reproducing the numbers

```
python -m experiments.run_matrix --op-sweep      # full Section-4 matrix + OP sweep
python -m ml.evaluation.cost                      # accuracy-vs-cost table
python -m ml.evaluation.plots                     # comparative figures
```
Every metrics row carries a `run_id` = hash of that cell's effective config.

---

## Core Concept: Hot/Cold vs. Valid/Invalid

An essential design principle in SmartGC is the independence of validity and temperature:
- **Valid vs. Invalid**: Represents current block state. A block is valid if it holds the most recent version of an LBA; it is invalid if it has been overwritten. Garbage collection decisions (which segments to clean and which blocks to migrate) are driven strictly by **validity**.
- **Hot vs. Cold**: Represents update frequency / temporal lifetime. Hot blocks are expected to be rewritten soon; cold blocks persist across long horizons. Hot/cold classification influences **placement** (which segment receives the write), isolating cold data from early-invalidating hot data to reduce future migrations.

---

## System Architecture & End-to-End Pipeline

The project is decoupled into two independent components communicating strictly via CSV files:

```
Raw I/O Trace  (synthetic Zipf  |  MSR/FIU real block trace  |  drift scenario)
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
│   ├── inference/            # rolling_cutoff, drift, confidence_gate, export.py
│   └── evaluation/           # cost.py (accuracy-vs-cost), plots.py
├── experiments/              # run_matrix.py — full Section-4 benchmark matrix
├── results/
│   ├── metrics/              # CSV output metrics from simulator runs
│   └── plots/                # Comparative visualizations
├── scripts/                  # Helper automation scripts
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
cmake -S simulator -B simulator/build
cmake --build simulator/build --config Release
```

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
