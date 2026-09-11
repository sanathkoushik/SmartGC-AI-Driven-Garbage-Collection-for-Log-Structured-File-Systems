# SmartGC Architecture & Interface Contracts

## 1. Architectural Philosophy

SmartGC decouples the deep learning sequence modeling components from the storage engine simulation. Rather than introducing language bindings (e.g., pybind11), shared memory IPC, or runtime network calls, the system uses a **file-based contract (CSV)**:

1. **Python ML Subsystem (`ml/`)**: Handles trace parsing, sequence generation, dataset splitting, PyTorch LSTM training, and batch offline inference. It exports predictions to standard CSV files.
2. **C++17 Simulator Subsystem (`simulator/`)**: A deterministic Log-Structured File System (LFS) and Greedy Garbage Collection engine. It consumes normalized traces and prediction files, simulates physical segment allocation and valid-data migration, and outputs performance metrics.

```
+-------------------------------------------------------------------+
|                        Python ML Subsystem                        |
|                                                                   |
| [Raw Trace] -> [Preprocessing] -> [Sequences] -> [LSTM Training]  |
|                                                         │         |
|                                                         ▼         |
|                                               [Inference Export]  |
+---------------------------------------------------------│---------+
                                                          │ CSV
                                                          ▼
+-------------------------------------------------------------------+
|                        C++ Simulator                              |
|                                                                   |
| [Trace Player] -> [LBA Classifier / Placement]                    |
|                          │                                        |
|                          ▼                                        |
| [Log-Structured Segment Allocation] <-> [Greedy GC Engine]        |
|                          │                                        |
|                          ▼                                        |
|                   [Metrics & WAF]                                 |
+-------------------------------------------------------------------+
```

---

## 2. CSV Interface Contracts

**`contract_version = 2`** (see `simulator/include/types.hpp::CONTRACT_VERSION` and
`ml/common/io_contracts.py`). Version 1 was the Phase 1 schema. Version 2 added
`predicted_stream_class` + `confidence` to predictions and the stream / learned-
trigger / tail-latency counters to the metrics row. Every metrics row now carries
a `contract_version` column so old and new CSVs are distinguishable.

### 2.1 Normalized Trace Contract (`data/processed/*.csv`)
Produced by `ml/preprocessing/normalize.py` (Phase 2), consumed by sequence
builders and the C++ simulator trace player (`smartgc_sim --trace`).

| Column Name | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `timestamp` | `uint64_t` | Normalized, monotonically non-decreasing logical I/O time (re-based to 0) | `100234` |
| `lba` | `uint64_t` | Logical Block Address of the request | `1048576` |
| `size` | `uint32_t` | Request size **in 4 KiB blocks** (integer, `>= 1`) | `1` |
| `operation` | `string` | Request type (`W` write, `R` read, `D` discard) | `W` |

**Header Format**: `timestamp,lba,size,operation`

`normalize.py` ingests `--source synthetic` (the `smartgc_sim --export-trace`
CSV), `--source spc` (UMass/SPC `ASU,LBA,Size,Opcode,Timestamp` rows — LBA is a
512-byte sector index per ASU, Size is bytes, Timestamp is elapsed seconds),
`--source msr` (MSR Cambridge FILETIME/byte-offset rows), or `--source auto`
(header-sniffed FIU/blktrace exports), with optional `--dense-lba`,
`--working-set` folding and `--max-events` truncation.

### 2.2 Predictions Contract (`data/predictions/*.csv`) — v2
Produced by `ml/inference/export.py` (Phase 4b), consumed by `simulator/` on
`--predictions`. Exactly **one row per write event, in trace order** (lockstep
with the replay); events with no usable history fall back to the RULE_BASED
value with low `confidence`.

| Column Name | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `timestamp` | `uint64_t` | Timestamp of the trace write event | `100234` |
| `lba` | `uint64_t` | Logical Block Address being written | `1048576` |
| `predicted_rewrite_interval` | `double` | Predicted write-events until the next overwrite | `142.5` |
| `predicted_class` | `string`/`uint8_t` | `HOT`=`1` / `COLD`=`0`, from the rolling percentile cutoff | `HOT` |
| `predicted_stream_class` | `uint8_t` | Multi-level migration bucket: `0`=SHORT, `1`=MEDIUM, `2`=LONG (Phase 5) | `1` |
| `confidence` | `double` | `[0,1]` predictive confidence; `< ml.confidence_fallback_threshold` ⇒ RULE_BASED fallback (Phase 4b) | `0.82` |

**Header Format**: `timestamp,lba,predicted_rewrite_interval,predicted_class,predicted_stream_class,confidence`

v1 files (first four columns only) are still accepted by
`simulator/include/predictions.hpp`: `predicted_stream_class` is then derived
from `predicted_class` and `confidence` defaults to `1.0`.

### 2.3 Simulator Metrics Contract (`results/metrics/*.csv`) — v2
Appended by `simulator/` at the end of every run (`--export-metrics`).

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `contract_version` | `int` | Schema version (`2`) |
| `run_id` | `string` | Hash of the cell's effective config (set by `experiments/run_matrix.py`), for reproducibility |
| `workload_name` | `string` | Trace / synthetic workload name |
| `placement_policy` | `string` | `MIXED`, `RULE_BASED`, `SUP_LIKE`, `STAT_ML`, `LSTM_SMARTGC`, `LSTM_ATTN_SMARTGC` |
| `total_segments`, `blocks_per_segment` | `size_t` | Storage geometry |
| `logical_bytes_written` | `uint64_t` | Bytes written by incoming workload requests |
| `physical_bytes_written` | `uint64_t` | Initial writes + GC migrations |
| `gc_bytes_copied` | `uint64_t` | Bytes migrated by GC |
| `valid_blocks_migrated` | `uint64_t` | Valid blocks migrated by GC |
| `gc_count` | `uint64_t` | GC cleaning operations |
| `waf` | `double` | `physical_bytes / logical_bytes` |
| `migration_stream_count` | `size_t` | Active write/GC-migration log heads (1–3) |
| `learned_trigger_enabled` | `0/1` | Whether the learned GC trigger was used |
| `avg_gc_decision_latency_us` | `double` | Mean wall time of a GC-trigger decision |
| `gc_migrated_p50/p95/p99` | `double` | Blocks migrated per GC invocation, distribution (tail-latency proxy, Phase 7) |
| `predicted_writes` | `uint64_t` | Writes that consulted a prediction row |
| `confidence_fallbacks` | `uint64_t` | Of those, how many fell back to RULE_BASED |
| `confidence_fallback_rate` | `double` | `confidence_fallbacks / predicted_writes` |

### 2.4 Learned GC-Trigger Q-table (`data/predictions/gc_policy_<name>.csv`)
Produced by `ml/training/gc_controller.py` (Phase 5), consumed by `simulator/`
on `--learned-trigger --gc-policy-table`.

**Header Format**: `state_id,q_wait,q_trigger`

`state_id` is the `0..26` discretisation of `[free_segment_ratio,
recent_write_rate, recent_waf-1]` at 3×3×3 buckets
(`GcTriggerController::discretize`). Only **visited** states are written; the C++
controller applies a deterministic adaptive rule for any absent state.

### 2.5 Auxiliary reports
- `results/metrics/trace_stats_<name>.csv` (+ `_hist.csv`) — Phase 2 workload
  statistics: unique LBAs, write/read ratio, rewrite-interval distribution,
  top-20%-LBA write share.
- `results/metrics/drift_status_<name>.json` — Phase 4b drift detector state:
  `needs_retrain`, `last_flag_event`, and a per-window KL timeline. Polled by
  `ml/training/retrain_now.py`.
- `ml/models/scaler.json` — Phase 3 fitted normalization (feature mean/std +
  `log1p` target transform). Inference reloads it and never refits.

### 2.6 Rolling percentile cutoff — update rule (Phase 4b)
`ml/inference/rolling_cutoff.py` replaces the fixed `ml.hot_percentile_cutoff`
constant. For each write event, in trace order:

1. `classify` / `stream_class` are evaluated against the percentile(s) of the
   intervals **currently** in the window (the last `ml.hot_cutoff_window_events`
   predictions, *before* this event is appended) — decisions use only
   information available at decision time.
2. The new predicted interval is then appended; the oldest is evicted once the
   window is full.
3. During warm-up (`< min_samples`) the seed constant `ml.hot_percentile_cutoff`
   is applied to the running list.

`HOT ⇔ interval ≤ P(hot_percentile_cutoff)`. For `N` migration streams the
SHORT/…/LONG bucket edges are the evenly-spaced `100·k/N` percentiles of the
window.

### 2.8 Phase 8 — HYBRID_ROBUST_SMARTGC: learning-augmented robustness blend
`ml/inference/robust_blend.py`, wired into `ml/inference/export.py`. Gap
addressed: the Section-4 ladder found that on the real UMass SPC Financial1
trace `RULE_BASED` beats every learned rung, and that the *existing* binary
`ConfidenceGate` still trusts the model on the majority of real events yet is
often wrong on that trusted majority. `HYBRID_ROBUST_SMARTGC` reuses the
`LSTM_ATTN_SMARTGC` checkpoint verbatim (`ml/models/registry.py` maps both
names to the same class; no separate training) but replaces the gate's hard
threshold with a continuous blend:

```
trust_weight = clip(confidence, 0, 1) * (1 - robustness_lambda)
blended_interval = trust_weight * model_interval + (1 - trust_weight) * rule_based_interval
```

`robustness_lambda = 0` recovers pure confidence-weighted trust in the model;
`robustness_lambda = 1` always uses the `RULE_BASED` interval regardless of
confidence. This directly instantiates the *consistency-robustness* tradeoff
from Lange, Naor & Yadgar, "Optimal SSD Management with Predictions" (ACM
SIGMETRICS 2025; `docs/related_work.md` row 9) as a tunable inference-time
policy — a heuristic instantiation of that framing for this empirical ladder,
not a reproduction of the paper's formal worst-case algorithm. The blended
interval feeds the same rolling-percentile cutoff as every other rung, so it
is a drop-in placement/migration policy: the simulator consumes it through the
identical `predicted_stream_class`/`confidence` contract, dispatched by the
same C++ code path as `LSTM_ATTN_SMARTGC` (`PlacementPolicy::HYBRID_ROBUST_SMARTGC`,
`simulator/include/types.hpp`). Because the Python layer has already folded
confidence into the interval, the simulator's own confidence-based re-gate is
disabled for this policy (`--confidence-threshold 0`) to avoid double-applying
two different heuristics. `robustness_lambda` is tuned empirically via
`python -m experiments.robustness_sweep` (`results/metrics/robustness_sweep.csv`,
`results/plots/robustness_tradeoff.png`); see `docs/progress.md` Phase 8 for
the resulting numbers.

**Note — two independent RULE_BASED heuristics.** `robustness_lambda = 1`
folds `HYBRID_ROBUST_SMARTGC` down to "always trust the RULE_BASED interval",
but this does **not** exactly reproduce the ladder's own `RULE_BASED` row, and
the discrepancy is a pre-existing property of the codebase worth stating
plainly: `PlacementPolicy::RULE_BASED` in the C++ simulator
(`simulator/src/lfs_simulator.cpp::rule_based_class()`) never reads
`predictions.csv` at all -- it computes its own zero-training heuristic
directly from `write_count_`/`last_write_event_` with a fixed
`blocks_per_segment`-sized threshold. Every other rung's Python-side fallback
(including the robustness blend here) instead goes through
`ml/models/rule_based.py::RuleBasedModel` -- a different heuristic (a blend of
the last observed gap and the rolling mean interval) whose output is then run
through the same dynamic rolling-percentile cutoff as the learned rungs. Both
are legitimate, independently-motivated zero-training baselines; they are
simply not the same computation, so `HYBRID_ROBUST_SMARTGC` at
`robustness_lambda=1` should be read as "always trust the *Python* RULE_BASED
signal, routed through the shared dynamic cutoff", not as an exact
re-derivation of the `RULE_BASED` ladder rung.

### 2.7 Real-trace source / attribution
The benchmark's real-trace slot (`evaluation.real_trace_name = "financial1"`)
uses the **UMass SPC Financial1** trace — OLTP block I/O from a financial
institution, distributed via the **UMass Trace Repository** (originally the
Storage Performance Council, SPC-1). Financial1 (5.33M events, **76.8% writes**)
and Financial2 (3.70M events, **17.7% writes**) were both inspected with
`trace_stats.py`; Financial2 is read-dominated and is **excluded** from the
matrix per the write-heavy-volume guidance. The `.spc` files are not vendored
(size); `run_matrix.py` normalizes a bounded `evaluation.real_trace_max_events`
prefix via `--source spc`. This dataset replaced an MSR-Cambridge / SNIA IOTTA
plan due to access availability, not methodology.
`ml/preprocessing/make_sample_trace.py` still emits an MSR-format **stand-in**
for reproducibility when no real trace is present; FIU / SNIA blktrace exports
work through `--source auto`.

---

## 3. LFS Simulator Core Engine Design

### 3.1 Block Lifecycle
Each block in a segment transitions through three states:
1. `FREE`: Unallocated block ready to receive data.
2. `VALID`: Contains the current, live version of an LBA.
3. `INVALID`: Holds an obsolete copy of an LBA after a subsequent overwrite elsewhere in the log.

$$\text{FREE} \xrightarrow{\text{write}} \text{VALID} \xrightarrow{\text{overwrite}} \text{INVALID} \xrightarrow{\text{GC reclaim}} \text{FREE}$$

### 3.2 Segment Model
A segment is a fixed contiguous array of $N$ blocks. Each segment maintains counters:
- `valid_count`: Number of active blocks currently mapped.
- `invalid_count`: Number of obsolete blocks.
- `free_count`: Number of available unwritten blocks.

### 3.3 Logical-to-Physical Mapping Table (L2P)
An in-memory hash table mapping `LBA -> PhysicalAddress(segment_id, block_index)`.
- On every write to an existing LBA, the previous physical block at `(prev_segment, prev_block)` is marked `INVALID`.
- The new block at the open segment tail is marked `VALID` with `lba`, and L2P is updated.

### 3.4 Greedy Garbage Collection
- **Trigger**: When free segments drop to or below the configured low watermark (`gc_trigger_watermark`).
- **Victim Selection**: Identifies closed (dirty) segments and selects the segment with the lowest `valid_count` (maximum invalid blocks).
- **Migration**: For each `VALID` block in the victim segment:
  - Read block LBA.
  - Write block sequentially into the active open segment.
  - Update L2P mapping.
  - Increment `gc_bytes_copied`, `valid_blocks_migrated`, and `physical_bytes_written`.
  - **Crucial rule**: GC migrations do **NOT** increment `logical_bytes_written`.
- **Reclamation**: The victim segment is completely erased: all blocks marked `FREE`, valid/invalid counters reset to 0, free counter reset to $N$, and returned to the free segment pool.

### 3.5 Multi-Level Migration Streams (Phase 5)
The engine keeps **`migration_stream_count` (1–3) concurrent open log heads**, one
per lifetime class `SHORT / MEDIUM / LONG` (`simulator/include/types.hpp::StreamClass`).

- **Placement time**: `decide_placement_stream()` routes a fresh write to a head
  by policy — `MIXED` → single head; `SUP_LIKE` → always `SHORT`; `RULE_BASED` →
  the recent-interval heuristic; `STAT_ML` / `LSTM_SMARTGC` → the binary
  `predicted_class`; `LSTM_ATTN_SMARTGC` → the `predicted_stream_class` bucket.
- **GC-migration time**: `decide_migration_stream()` re-routes each surviving
  valid block using the *most recent prediction on record for that LBA*
  (`last_pred_stream_`), so lifetime separation is applied again at copy-forward,
  not only at first write (Shiro-style multi-destination migration). `SUP_LIKE`
  sends every GC survivor to `LONG`.
- `phys_stream()` collapses the semantic class onto an actual head index for the
  current stream count (2 heads ⇒ SHORT-vs-rest; 3 heads ⇒ identity).
- **Safety reserve**: with `N` heads a single GC pass can transiently consume up
  to `N` fresh segments for migration destinations, so multi-stream runs keep a
  reserve of `max(N+1, gc_free_segments_threshold)` free segments and, on
  trigger, clean repeatedly until the pool is back above that reserve. The
  single-stream path is byte-identical to Phase 1 (baseline WAF `1.0544`).

### 3.6 Learned GC Trigger (Phase 5, ablatable)
When `gc.learned_trigger: true`, the fixed low-watermark check is replaced by
`GcTriggerController` (`simulator/include/gc_trigger.hpp`):

- **State** `[free_segment_ratio, recent_write_rate, recent_waf]`, discretised
  3×3×3 → `state_id ∈ [0,26]`. `recent_write_rate` = user writes since the last
  GC decision / `gc.trigger_state_window_events`, clamped to `[0,1]`.
- **Action** `{wait, trigger_now}` = `argmax` of the loaded Q-table row; for any
  `state_id` absent from `gc_policy_<name>.csv` a deterministic adaptive rule is
  used (raise the effective watermark during write bursts / rising WAF, relax it
  when the system is calm). The pool-safety floor still forces a trigger at
  `free ≤ streams`.
- `avg_gc_decision_latency_us` is measured around every decision and reported.
- `gc.learned_trigger: false` (default) reproduces the Phase 1 fixed watermark
  exactly, so the baseline always remains runnable.

---

## 4. Write Amplification (WAF) Formula

$$\text{WAF} = \frac{\text{Physical Bytes Written}}{\text{Logical Bytes Written}} = \frac{\text{Logical Bytes Written} + \text{GC Migrated Bytes}}{\text{Logical Bytes Written}} = 1 + \frac{\text{GC Migrated Bytes}}{\text{Logical Bytes Written}}$$

If no GC runs, $\text{WAF} = 1.0$. As GC migrates valid blocks, $\text{WAF} > 1.0$. Smart hot/cold separation aims to minimize $\text{GC Migrated Bytes}$, thereby driving WAF closer to 1.0.
