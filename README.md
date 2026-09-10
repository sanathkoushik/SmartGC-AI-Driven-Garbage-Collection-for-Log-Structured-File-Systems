# SmartGC: AI-Driven Garbage Collection for Log-Structured File Systems

## Project Overview

In a Log-Structured File System (LFS), data is written sequentially to log segments rather than updated in place. As files are modified and deleted over time, storage segments accumulate a mixture of obsolete (**invalid**) and active (**valid**) blocks. To reclaim storage space, a Garbage Collector (GC) must identify victim segments, copy their remaining valid blocks to the head of the log (valid-data migration), and erase the segment for reuse.

When a segment co-locates **hot** data (frequently rewritten) and **cold** data (rarely rewritten), the hot blocks invalidate rapidly while cold blocks remain valid. When GC reclaims such mixed segments, it is forced to repeatedly copy the surviving cold data forward. This repeated copying of valid data directly drives **Write Amplification (WAF)**:

$$\text{WAF} = \frac{\text{Physical Bytes Written}}{\text{Logical Bytes Written}}$$

**SmartGC** investigates whether a sequence-based deep learning model (LSTM) trained on historical block-level rewrite intervals can predict future rewrite intervals accurately enough to classify logical block addresses (LBAs) as **HOT** or **COLD** *before* placement, and whether that improved prediction actually translates into fewer valid-block migrations and lower WAF.

---

## Honest Scope & Academic Novelty Framing

Hot/cold data separation itself is **not novel** — production file systems (such as Linux F2FS) and the flash-translation-layer literature already separate data by temperature.

SmartGC's specific, modest research goal is:
> To evaluate whether a lightweight LSTM trained on real block-I/O rewrite history captures temporal rewrite patterns better than simple threshold heuristics, whether pretraining on one set of workloads and fine-tuning on a held-out workload beats training from scratch, and whether any of that improvement survives the trip through a deterministic LFS simulator to become a measurable WAF reduction.

The comparison is:
1. **MIXED** — standard LFS; all writes enter a single active segment.
2. **RULE_BASED** — a simple threshold on recent rewrite intervals.
3. **LSTM from scratch** — trained only on the target workload.
4. **Pretrained LSTM** — trained on other workloads, applied without adaptation.
5. **Pretrained + fine-tuned LSTM** — the full SmartGC pipeline.

**A negative or mixed result is a valid outcome.** If the LSTM predicts better but does not reduce WAF, or if the rule-based heuristic wins, that is what will be reported.

---

## Core Concept: Hot/Cold vs. Valid/Invalid

An essential design principle in SmartGC is the independence of validity and temperature:

- **Valid vs. Invalid** — current block state. A block is valid if it holds the most recent version of an LBA; invalid once overwritten. Garbage collection decisions (which segment to clean, which blocks to migrate) are driven **strictly by validity**.
- **Hot vs. Cold** — predicted update frequency. This influences **placement only** (which append point receives the write).

A mispredicted temperature can cost write amplification; it can never lose or corrupt data. This is asserted directly by `test_temperature_routes_placement_only`.

---

## Datasets

SmartGC is evaluated on **real, public block-level I/O traces**. No conference
result is produced from synthetic data; synthetic workloads exist only for unit
tests and development, and every simulator run is labelled with the dataset it
used.

Both datasets download **anonymously over HTTPS with no form, no account and no
credentials**:

```bash
python scripts/download_datasets.py --dataset all
python scripts/verify_datasets.py
```

### Primary — MSR Cambridge (2007)

One-week block I/O traces of enterprise servers at Microsoft Research Cambridge:
36 volumes across 13 servers, per the distribution README. Records are

```
Timestamp,Hostname,DiskNumber,Type,Offset,Size,ResponseTime
```

with the timestamp in Windows FILETIME units, the offset in bytes from the start
of the logical disk, and the size in bytes.

* **Original data:** Microsoft Research Cambridge.
* **Citation:** Narayanan, Donnelly and Rowstron, "Write Off-Loading: Practical
  Power Management for Enterprise Storage", USENIX FAST '08.
* **Canonical repository:** <https://iotta.snia.org/traces/block-io/388>.
* **Downloaded from:** the public `cache-datasets` S3 bucket published by the
  cacheMon group, which holds the **original distribution archives unmodified**.
* **Integrity:** every volume is verified against `MD5.txt` — the checksum
  manifest **written by the data's authors** and shipped inside the archive.

### External generalization — SYSTOR '17 (2016)

Enterprise virtual desktop infrastructure traces from Fujitsu Laboratories,
2,706 hourly traces across 6 LUNs. Records are
`Timestamp,Response,IOType,LUN,Offset,Size`.

* **Citation:** Lee, Kumano, Matsuki, Endo, Fukumoto and Sugawara,
  "Understanding storage traffic characteristics on enterprise virtual desktop
  infrastructure", SYSTOR '17.
* **Canonical repository:** <https://iotta.snia.org/traces/block-io/4928>.
* **Downloaded from:** the same public bucket.

SYSTOR is used to test whether a model pretrained on MSR transfers to a
**different workload family, nine years newer** — a stronger generalization test
than another workload from the same era.

### Why not FIU

FIU traces were the intended secondary dataset and are **not obtainable
automatically**. All four routes were probed directly on 2026-09-09: SNIA IOTTA
and the Harvey Mudd mirror both gate bulk downloads behind a form requiring a
name, e-mail address and organisation; the original FIU host
(`sylab-srv.cs.fiu.edu`) does not respond; and every download link on the ASU
VISA Lab trace pages returns HTTP 404.

The FIU parser is implemented and unit-tested against real FIU sample records, so
files placed in `data/raw/fiu/` are picked up with no code change. The full
record, including the scored comparison of nine candidate sources, is in
[`docs/dataset_source_decision.md`](docs/dataset_source_decision.md) and
[`results/dataset_source_comparison.csv`](results/dataset_source_comparison.csv).

### Honest caveats

* **These traces are old.** MSR Cambridge was collected in **2007** and
  SYSTOR '17 in **2016**. SNIA files both under "Historical Traces". They remain
  the standard public block-I/O corpora for this kind of study, but no claim is
  made that they represent contemporary storage workloads.
* **The download host is not the data's producer**, and this repository never
  says otherwise.
* **No trace data is committed.** Microsoft's `DISCLAIMER.txt` reserves
  reproduction rights, so the pipeline reproduces the data from documented URLs
  instead.

Full provenance, field semantics and preprocessing: [`data/README.md`](data/README.md)
and [`docs/dataset.md`](docs/dataset.md).

## System Architecture & End-to-End Pipeline

```
Real block-I/O trace (MSR Cambridge / SYSTOR '17)
      |
      v
[Preprocessing & normalization]  -> timestamp,lba,size,operation,trace_id
      |
      v
[Rewrite-interval extraction]    -> per-LBA gaps between consecutive writes
      |
      v
[Chronological sequence build]   -> train / validation / test, no shuffling
      |
      v
[LSTM: pretrain -> fine-tune]    -> predicted next rewrite interval
      |
      v
[Hot/cold threshold]             -> derived from TRAIN data only
      |
      v  predictions.csv
[C++17 LFS Simulator]
      |
      +-- MIXED           (baseline: one append point)
      +-- RULE_BASED      (heuristic hot/cold placement)
      +-- LSTM_SMARTGC    (predicted hot/cold placement)
      |
      v  metrics.csv
[WAF, valid-block migrations, GC invocations, physical writes]
```

Python and C++ communicate **only through CSV files** (contracts in `docs/architecture.md` §2) — no language bindings, no IPC, no runtime coupling.

---

## Repository Structure

```
SmartGC/
├── README.md
├── requirements.txt          # Python dependencies
├── config/config.yaml        # Central configuration, read by both subsystems
├── data/
│   ├── raw/                  # Downloaded traces (gitignored; multi-GB)
│   ├── processed/            # Normalized traces & sequences (gitignored)
│   └── predictions/          # Per-write model predictions (gitignored)
├── models/
│   ├── pretrained/           # MSR-pretrained checkpoints + metadata
│   └── finetuned/            # Workload-adapted checkpoints + metadata
├── simulator/
│   ├── CMakeLists.txt        # Authoritative build definition
│   ├── include/              # Block, Segment, Mapping, Metrics, Config, LfsSimulator
│   ├── src/                  # Implementation & CLI runner
│   └── tests/                # Unit test suite
├── ml/
│   ├── preprocessing/        # Trace parsing, normalization, sequence building
│   ├── models/               # PyTorch model definitions
│   ├── training/             # Pretraining & fine-tuning loops
│   ├── inference/            # Prediction export
│   └── evaluation/           # Regression & classification metrics
├── experiments/              # Experiment orchestration
├── scripts/                  # Dataset download, build helper, conference demo
├── results/                  # Committed: metrics, statistics, plots, final tables
└── docs/                     # architecture.md, dataset.md, progress.md, ...
```

---

## Building and Running the C++ Simulator

### Prerequisites
- A C++17 compiler (GCC 9+, Clang 10+, or MSVC 2019+)
- CMake 3.15+

### Build

```bash
python scripts/build_simulator.py --test
```

The helper locates CMake (including a `winget` install that is not yet on PATH), picks a working generator, builds, and runs the unit tests. It is a thin wrapper — `simulator/CMakeLists.txt` is authoritative, and the plain commands work too:

```bash
cmake -S simulator -B simulator/build -DCMAKE_BUILD_TYPE=Release
cmake --build simulator/build --config Release
```

> On MSYS2/MinGW the build links the GCC runtime statically. This is required, not cosmetic: some MSYS2 installations cannot link the shared runtime at all — `ld` exits with status 116 and no diagnostic for any code that uses exceptions.

### Run the tests

```bash
./simulator/build/simulator_tests
```

Run from the repository root so the suite can validate `config/config.yaml`.

### Run the simulator

```bash
./simulator/build/smartgc_sim --config config/config.yaml
./simulator/build/smartgc_sim --help
```

Useful options: `--utilization 0.75` sets the working set as a fraction of raw capacity; `--placement MIXED|RULE_BASED|LSTM_SMARTGC`; `--gc-reserved N`; `--export-metrics results/metrics/run.csv`.

Exit codes: `0` success, `1` internal error, `2` bad arguments or configuration, `3` device full / working set exceeds usable capacity.

---

## Python Environment

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -r requirements.txt
```

(Use `.venv/bin/python` on Linux and macOS.) The models are small enough to train on CPU; a GPU is used if present but is not required.

---

## Key Simulator Semantics

- **Over-provisioning is explicit.** `simulator.gc_reserved_segments` (>= 1) reserves segments only the collector may consume. A victim segment is never fully valid, so it holds at most `blocks_per_segment - 1` survivors and one reserved segment always absorbs a cleaning pass. Usable capacity is derived from this, and a write that would exceed it raises `DeviceFullError` rather than failing obscurely later.
- **Three append points.** `MAIN` (all user writes under MIXED, hot writes otherwise), `COLD` (predicted-cold writes), and `GC` (migrated survivors). Keeping migrations out of the user log stops survivors being re-interleaved with short-lived data. `gc_separate_stream: false` folds them back in, retained as an ablation.
- **One record, one block.** `write()` rejects any request that is not exactly one block; multi-block requests are expanded during preprocessing so a 128 KB request is never accounted as 4 KB.
- **Fair comparison.** Every policy replays the identical normalized trace against the identical geometry, working set and seed. Only the placement decision differs.

Measured foundation results, including the utilization sweep and the effect of GC stream separation, are in `results/metrics/foundation_validation.csv` and `docs/progress.md`.
