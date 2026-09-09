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
|                                                         |         |
|                                                         v         |
|                                               [Inference Export]  |
+---------------------------------------------------------|---------+
                                                          | CSV
                                                          v
+-------------------------------------------------------------------+
|                        C++ Simulator                              |
|                                                                   |
| [Trace Player] -> [LBA Classifier / Placement]                    |
|                          |                                        |
|                          v                                        |
| [Log-Structured Segment Allocation] <-> [Greedy GC Engine]        |
|                          |                                        |
|                          v                                        |
|                   [Metrics & WAF]                                 |
+-------------------------------------------------------------------+
```

---

## 2. CSV Interface Contracts

### 2.1 Normalized Trace Contract (`data/processed/normalized/*.csv`)
Produced by `ml/preprocessing/` (Phase 2), consumed by sequence builders and the C++ simulator trace player.

| Column Name | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `timestamp` | `uint64_t` | Normalized, monotonically non-decreasing I/O timestamp | `100234` |
| `lba` | `uint64_t` | Logical Block Address, in units of `block_size_bytes` | `1048576` |
| `size` | `uint32_t` | Request size **in blocks**, never bytes | `1` |
| `operation` | `char` | `W` write, `R` read, `D` discard | `W` |
| `trace_id` | `string` | Identifier of the source workload | `msr_hm_0` |

**Header Format**:
```csv
timestamp,lba,size,operation,trace_id
```

**Block expansion rule.** A raw request that spans several blocks is expanded during preprocessing into one record per block, so every emitted record has `size == 1`. A 128 KB request at a 4 KB block size becomes 32 consecutive records. The simulator enforces this: `LfsSimulator::write` rejects any request whose byte length is not exactly one block, so a multi-block request can never be silently accounted as a single block.

### 2.2 Predictions Contract (`data/predictions/*.csv`)
Produced by `ml/inference/`, consumed by the simulator during smart-placement runs.

| Column Name | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `timestamp` | `uint64_t` | Timestamp of the write event this prediction applies to | `100234` |
| `lba` | `uint64_t` | Logical Block Address being written | `1048576` |
| `predicted_rewrite_interval` | `double` | Predicted time until the next overwrite of this LBA | `142.5` |
| `predicted_class` | `string` | `HOT` or `COLD` | `HOT` |
| `model_version` | `string` | Checkpoint identity (name + training stage) | `msr_pretrained_ft_prxy0` |
| `trace_id` | `string` | Source workload, matching the normalized trace | `msr_prxy_0` |

**Header Format**:
```csv
timestamp,lba,predicted_rewrite_interval,predicted_class,model_version,trace_id
```

Predictions are emitted only for write events that have enough rewrite history to fill a full input window. Write events without a prediction are passed to the simulator with `Temperature::UNKNOWN` and counted in `unpredicted_writes`, so prediction coverage is always measured rather than assumed.

### 2.3 Simulator Metrics Contract (`results/metrics/*.csv`)
Produced by `simulator/` at the conclusion of every simulation run.

| Column | Description |
| :--- | :--- |
| `dataset` | `msr`, `fiu` or `synthetic` |
| `trace` | Workload name |
| `policy` | `MIXED`, `RULE_BASED` or `LSTM_SMARTGC` |
| `model_type` | `none`, `rule`, `scratch`, `pretrained`, `pretrained_finetuned` |
| `total_segments`, `blocks_per_segment`, `block_size_bytes` | Device geometry |
| `gc_reserved_segments`, `gc_separate_stream` | Cleaning configuration |
| `target_utilization` | Live working set as a fraction of raw capacity |
| `seed` | Random seed of the run |
| `total_write_requests` | User write requests replayed |
| `logical_bytes_written` | Bytes written by the workload (never includes migrations) |
| `physical_bytes_written` | Bytes reaching the media (user writes + migrations) |
| `gc_bytes_copied` | Bytes migrated during cleaning |
| `valid_blocks_migrated` | Valid blocks relocated during cleaning |
| `gc_count` | Cleaning passes performed |
| `main_stream_writes`, `cold_stream_writes` | Placement split |
| `unpredicted_writes` | User writes that had no prediction available |
| `waf` | `physical_bytes_written / logical_bytes_written` |

---

## 3. LFS Simulator Core Engine Design

### 3.1 Block Lifecycle
Each block in a segment transitions through three states:
1. `FREE`: Unallocated block ready to receive data.
2. `VALID`: Contains the current, live version of an LBA.
3. `INVALID`: Holds an obsolete copy of an LBA after a subsequent overwrite elsewhere in the log.

```
FREE --write--> VALID --overwrite--> INVALID --GC reclaim--> FREE
```

### 3.2 Segment Model
A segment is a fixed contiguous array of *N* blocks with counters `valid_count`, `invalid_count` and `free_count`, which always sum to the segment capacity.

### 3.3 Logical-to-Physical Mapping Table (L2P)
An in-memory hash table mapping `LBA -> PhysicalAddress(segment_id, block_index)`.
- On every write to an existing LBA, the previous physical block is marked `INVALID`.
- The new block at an append point is marked `VALID` and the L2P entry is updated.

### 3.4 Append Points (Write Streams)

The log has three independent append points. Separating them is what makes hot/cold placement meaningful; it does not change validity semantics in any way.

| Stream | Carries |
| :--- | :--- |
| `MAIN` | Every user write under `MIXED`; predicted-**HOT** user writes under the segregated policies |
| `COLD` | Predicted-**COLD** user writes; never opened under `MIXED` |
| `GC` | Blocks migrated out of victim segments |

The `GC` stream exists so that surviving (by definition longer-lived) data is not interleaved into a user log, where it would be re-migrated on the next pass. Setting `simulator.gc_separate_stream: false` folds migrations back into `MAIN`, which is retained as an ablation.

A write with `Temperature::UNKNOWN` falls back to `MAIN`. This is the conservative choice: it never strands unclassified data in the cold segment.

### 3.5 Greedy Garbage Collection
- **Trigger**: A user write that needs a fresh segment first cleans until the free pool holds strictly more than `max(gc_free_segments_threshold, gc_reserved_segments)` segments.
- **Victim selection**: Among segments that are (a) not an open append point, (b) not empty, and (c) **not fully valid**, choose the one with the fewest valid blocks; ties break toward more invalid blocks. Excluding fully valid segments matters: cleaning one would migrate an entire segment's worth of data and free nothing.
- **Migration**: Each `VALID` block in the victim is appended to the `GC` stream and the L2P is updated. Migration increments `physical_bytes_written`, `gc_bytes_copied` and `valid_blocks_migrated`, and **never** `logical_bytes_written`.
- **Reclamation**: The victim is erased — all blocks `FREE`, counters reset — and returned to the free pool.

### 3.6 Over-Provisioning Reserve

`simulator.gc_reserved_segments` (>= 1) is a pool of segments that only the garbage collector may consume. It is what makes a cleaning pass always completable:

> A victim is never fully valid, so it holds at most `blocks_per_segment - 1` surviving blocks. One reserved segment therefore always absorbs one pass, and the victim is returned to the pool immediately afterwards.

Because a pass frees a whole segment while consuming at most `capacity - 1` blocks, **total free blocks strictly increase on every pass**, which also bounds the cleaning loop.

Without this reserve, a user write and the migrations it triggers compete for the same free segments and the pool can be exhausted mid-migration. That was an actual failure mode of the first implementation: at 95% utilization the engine aborted with "No free segment available to receive migrated valid block".

### 3.7 Admission Control

```
unavailable_segments = gc_reserved_segments + user_append_points + gc_append_point
max_live_blocks      = (total_segments - unavailable_segments) * blocks_per_segment
```

A write that would introduce a **new** LBA beyond `max_live_blocks` raises `DeviceFullError` with the live count and the limit, instead of failing later inside the collector. Overwrites of already-live LBAs are always admitted, since they do not grow the live set.

Note that `user_append_points` is 1 under `MIXED` and 2 under the segregated policies, so a segregated run has one segment less of usable capacity. Experiments that compare policies must therefore fix the geometry and the working set across all policies (see §4).

---

## 4. Write Amplification (WAF)

```
WAF = physical_bytes_written / logical_bytes_written
    = (logical_bytes_written + gc_bytes_copied) / logical_bytes_written
    = 1 + gc_bytes_copied / logical_bytes_written
```

If no cleaning runs, WAF = 1.0. Hot/cold separation aims to reduce `gc_bytes_copied` by concentrating short-lived blocks in segments that invalidate wholesale.

**WAF is only comparable across runs that share a device geometry, working set and request stream.** Every policy comparison in this project replays the identical normalized trace against the identical configuration; only the placement decision differs.

---

## 5. Hot/Cold is not Valid/Invalid

This separation is the central correctness property of the design:

| | Determined by | Affects |
| :--- | :--- | :--- |
| **Valid / Invalid** | Whether the block holds the newest version of its LBA | Which blocks GC migrates; when a segment can be erased |
| **Hot / Cold** | Predicted time until the next overwrite | Which append point receives the write |

A block predicted `COLD` is invalidated by its next overwrite exactly like a `HOT` one; a mispredicted temperature can cost write amplification but can never corrupt data or lose an LBA. `test_temperature_routes_placement_only` asserts this directly.
