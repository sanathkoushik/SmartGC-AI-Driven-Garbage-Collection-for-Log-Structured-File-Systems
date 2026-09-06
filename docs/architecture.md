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

### 2.1 Normalized Trace Contract (`data/processed/*.csv`)
Produced by `ml/preprocessing/` (Phase 2), consumed by sequence builders and the C++ simulator trace player.

| Column Name | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `timestamp` | `uint64_t` or `double` | Logical or wall-clock I/O timestamp (normalized monotonically non-decreasing) | `100234` |
| `lba` | `uint64_t` | Logical Block Address of the request | `1048576` |
| `size` | `uint32_t` | Request size in sectors or blocks (standardized to 4KB blocks) | `1` |
| `operation` | `string` or `uint8_t` | Request type (`W` for write, `R` for read, `D` for discard) | `W` |

**Header Format**:
```csv
timestamp,lba,size,operation
```

### 2.2 Predictions Contract (`data/predictions/*.csv`)
Produced by `ml/inference/` (Phase 4 & 5), consumed by `simulator/` during smart placement runs.

| Column Name | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `timestamp` | `uint64_t` or `double` | Timestamp matching the original trace write event | `100234` |
| `lba` | `uint64_t` | Logical Block Address being written | `1048576` |
| `predicted_rewrite_interval` | `double` | Predicted time / write count until next overwrite | `142.5` |
| `predicted_class` | `string` / `uint8_t` | Temperature label (`HOT` = `1`, `COLD` = `0`) | `HOT` |

**Header Format**:
```csv
timestamp,lba,predicted_rewrite_interval,predicted_class
```

### 2.3 Simulator Metrics Contract (`results/metrics/*.csv`)
Produced by `simulator/` at the conclusion of every simulation run.

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `workload_name` | `string` | Name of trace or synthetic workload |
| `placement_policy` | `string` | `MIXED` (baseline), `RULE_BASED`, `LSTM_SMARTGC` |
| `total_segments` | `size_t` | Storage capacity in segments |
| `blocks_per_segment` | `size_t` | Blocks per segment |
| `logical_bytes_written`| `uint64_t` | Total bytes written by incoming workload requests |
| `physical_bytes_written`| `uint64_t` | Total bytes written to flash (initial writes + GC migrations) |
| `gc_bytes_copied` | `uint64_t` | Bytes migrated by GC during segment reclamation |
| `valid_blocks_migrated`| `uint64_t` | Number of valid blocks migrated during GC |
| `gc_count` | `uint64_t` | Total number of GC cleaning operations performed |
| `waf` | `double` | Write Amplification Factor ($\text{physical\_bytes} / \text{logical\_bytes}$) |

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

---

## 4. Write Amplification (WAF) Formula

$$\text{WAF} = \frac{\text{Physical Bytes Written}}{\text{Logical Bytes Written}} = \frac{\text{Logical Bytes Written} + \text{GC Migrated Bytes}}{\text{Logical Bytes Written}} = 1 + \frac{\text{GC Migrated Bytes}}{\text{Logical Bytes Written}}$$

If no GC runs, $\text{WAF} = 1.0$. As GC migrates valid blocks, $\text{WAF} > 1.0$. Smart hot/cold separation aims to minimize $\text{GC Migrated Bytes}$, thereby driving WAF closer to 1.0.
