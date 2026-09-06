# SmartGC: AI-Driven Garbage Collection for Log-Structured File Systems

## Project Overview

In a Log-Structured File System (LFS), data is written sequentially to log segments rather than updated in place. As files are modified and deleted over time, storage segments accumulate a mixture of obsolete (**invalid**) and active (**valid**) blocks. To reclaim storage space, a Garbage Collector (GC) must identify victim segments, copy their remaining valid blocks to the head of the log (valid-data migration), and erase the segment for reuse. 

When a segment co-locates **hot** data (frequently rewritten) and **cold** data (rarely rewritten), the hot blocks invalidate rapidly while cold blocks remain valid. When GC reclaims such mixed segments, it is forced to repeatedly copy the surviving cold data forward. This repeated copying of valid data directly drives **Write Amplification (WAF)**:

$$\text{WAF} = \frac{\text{Physical Bytes Written}}{\text{Logical Bytes Written}}$$

**SmartGC** investigates whether a sequence-based deep learning model (LSTM) trained on historical block-level rewrite intervals can accurately predict future rewrite intervals and classify logical block addresses (LBAs) into **HOT** or **COLD** streams prior to placement. By placing predicted hot and cold blocks into segregated segments, SmartGC aims to minimize valid-data migration during garbage collection.

---

## Honest Scope & Academic Novelty Framing

Hot/cold data separation itself is **not novel** — modern production file systems (such as Linux F2FS) and extensive flash-translation layer (FTL) literature already separate data by temperature. 

SmartGC's specific, modest research goal is:
> To evaluate whether a lightweight LSTM sequence model trained on block-I/O rewrite history can capture temporal rewrite patterns more effectively than simple threshold heuristics, and to feed those predictions into a controlled, deterministic LFS simulator to directly measure the resulting impact on valid-data migration and WAF against two baselines:
> 1. **Baseline Greedy GC (Mixed Placement)**: Standard naive LFS where all writes enter a single active segment.
> 2. **Rule-Based Heuristic**: A simple threshold rule on recent/average rewrite intervals.
> 3. **SmartGC (LSTM-driven)**: Segment placement guided by predicted rewrite intervals.

---

## Core Concept: Hot/Cold vs. Valid/Invalid

An essential design principle in SmartGC is the independence of validity and temperature:
- **Valid vs. Invalid**: Represents current block state. A block is valid if it holds the most recent version of an LBA; it is invalid if it has been overwritten. Garbage collection decisions (which segments to clean and which blocks to migrate) are driven strictly by **validity**.
- **Hot vs. Cold**: Represents update frequency / temporal lifetime. Hot blocks are expected to be rewritten soon; cold blocks persist across long horizons. Hot/cold classification influences **placement** (which segment receives the write), isolating cold data from early-invalidating hot data to reduce future migrations.

---

## System Architecture & End-to-End Pipeline

The project is decoupled into two independent components communicating strictly via CSV files:

```
Raw I/O Trace
      │
      ▼
[Phase 2: Preprocessing & Normalization]
      │
      ▼ (normalized_trace.csv: timestamp, lba, size, operation)
[Phase 3: Sequence Construction & Split]
      │
      ▼
[Phase 4: Lightweight LSTM Model (PyTorch)]
      │
      ▼ (predictions.csv: timestamp, lba, predicted_interval, predicted_class)
[Phase 5 & 6: C++17 LFS Simulator]
      │
      ├── Baseline: Greedy GC (Mixed Placement)
      ├── Baseline: Rule-Based Hot/Cold Heuristic Placement
      └── SmartGC: LSTM-Guided Hot/Cold Placement
      │
      ▼ (metrics.csv)
[Phase 7: Evaluation & Comparison]
      │
      └── WAF, Valid Migration Count, Physical Writes, GC Invocations
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
│   ├── preprocessing/        # Trace parsing and normalization
│   ├── models/               # PyTorch sequence model architectures
│   ├── training/             # Training loop, loss functions, validation
│   ├── inference/            # Prediction export routines
│   └── evaluation/           # Sequence prediction metrics (MAE, RMSE)
├── experiments/              # Experiment orchestration scripts
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

### Running Baseline Simulator CLI
```bash
./simulator/build/smartgc_sim --help
```
