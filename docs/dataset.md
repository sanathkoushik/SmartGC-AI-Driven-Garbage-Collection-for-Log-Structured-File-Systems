# Datasets: what they are and exactly what is done to them

Provenance and how to obtain the data: [`data/README.md`](../data/README.md).
Why these sources rather than others: [`docs/dataset_source_decision.md`](dataset_source_decision.md).
This document covers **field semantics and every transformation applied**.

---

## 1. MSR Cambridge — primary

### What it is

Block-level I/O traces captured below the file-system cache on enterprise
servers at Microsoft Research Cambridge, over one week beginning in February
2007. The distribution README states: *"There are 36 I/O traces from 36
different volumes on 13 servers."* Each file is one logical volume, named
`<hostname>_<disknumber>.csv.gz`, and the hostnames identify the server's role —
`hm` hardware monitoring, `mds` media server, `prn` print server, `proj` project
directories, `prxy` web proxy, `rsrch` research projects, `src` source control,
`stg` web staging, `ts` terminal server, `usr` user home directories, `wdev` test
web server, `web` web/SQL server.

The SNIA catalogue records the collection as 434 million records / 4.94 GB across
two archives (271 M / 3.06 GB and 163 M / 1.88 GB).

### Who collected it, and how to cite it

Dushyanth Narayanan, Austin Donnelly and Antony Rowstron, *Write Off-Loading:
Practical Power Management for Enterprise Storage*, USENIX FAST '08. The
distribution README asks that this paper be cited by any published work using
the traces.

### Record format

```
Timestamp,Hostname,DiskNumber,Type,Offset,Size,ResponseTime
```

| Field | Meaning (quoted from the distribution README) |
| :--- | :--- |
| `Timestamp` | "the time the I/O was issued in *Windows filetime*" |
| `Hostname` | "the hostname (should be the same as that in the trace file name)" |
| `DiskNumber` | "the disknumber (should be the same as in the trace file name)" |
| `Type` | `"Read"` or `"Write"` |
| `Offset` | "the starting offset of the I/O in bytes from the start of the logical disk" |
| `Size` | "the transfer size of the I/O request in bytes" |
| `ResponseTime` | "the time taken by the I/O to complete, in Windows filetime units" |

A **Windows FILETIME** is a count of 100-nanosecond ticks since 1601-01-01 UTC.

### Why it suits SmartGC

The three things this project needs are all present and were confirmed by parsing
real records, not by reading the documentation:

1. **A logical block address space.** `Offset` is a byte position within one
   volume, so it maps directly onto fixed-size blocks.
2. **Reads distinguished from writes.** Only writes allocate blocks and drive
   cleaning in a log-structured store.
3. **Repeated writes to the same block.** Without these there are no rewrite
   intervals and the research question is undefined. Measured rewrite ratios per
   workload are in `results/dataset_inventory.csv`.

---

## 2. SYSTOR '17 — external generalization

### What it is

Block-level traces of an enterprise **virtual desktop infrastructure** deployment
at Fujitsu, collected February–March 2016. The distribution README states:
*"There are 2706 I/O traces from 6 different block storage devices (LUNs
0,1,2,3,4, and 6)."* Files are named `<YYYYMMDDHH>-LUN<n>.csv.gz` — one hour, one
LUN. Local time is UTC+0900 (JST).

### Who collected it, and how to cite it

Chunghan Lee, Tatsuo Kumano, Tatsuma Matsuki, Hiroshi Endo, Naoto Fukumoto and
Mariko Sugawara, *Understanding storage traffic characteristics on enterprise
virtual desktop infrastructure*, SYSTOR '17.

### Record format

```
Timestamp,Response,IOType,LUN,Offset,Size
```

| Field | Meaning |
| :--- | :--- |
| `Timestamp` | Unix time in seconds with a fractional part. The README notes the fraction has nine digits but is *"accurate only to the microsecond level; please ignore the nanosecond part."* |
| `Response` | time to complete the I/O |
| `IOType` | `R`, `W`, or **blank** |
| `LUN` | logical unit number — the device identity |
| `Offset` | byte offset |
| `Size` | byte count |

### Why this rather than another MSR workload

The question is whether a model pretrained on one workload family transfers to
another. SYSTOR '17 is nine years newer than MSR and a genuinely different
setting — hundreds of virtual desktops against shared LUNs, rather than
individual enterprise servers. It was chosen after FIU proved unobtainable (see
the source decision), and it is the stronger test of the two.

---

## 3. FIU — parser present, data absent

FIU's IODedup and SRCMap sets were the intended secondary dataset. The parser is
implemented and unit-tested against **real FIU sample records** (which are
downloadable without the licence form), but the bulk data could not be obtained:
SNIA IOTTA and the Harvey Mudd mirror gate it behind a personal-details form, the
original FIU host is offline, and the ASU VISA Lab links return 404.

Record format, from the SNIA README for trace 391:

```
[ts in ns] [pid] [process] [lba] [size in 512 Bytes blocks] [Write or Read]
[major device number] [minor device number] [MD5 per 4096 Bytes]
```

`lba` counts 512-byte sectors and `size` counts 512-byte blocks, so an 8-sector
request is exactly one 4 KB block.

---

## 4. Preprocessing, transformation by transformation

Implemented in `ml/preprocessing/`: one reader per dataset over a shared base
(`base_parser.py`) that owns everything below. Run with:

```bash
python -m ml.preprocessing.normalize --discover
```

### 4.1 The target contract

```
timestamp,lba,size,operation,trace_id
```

### 4.2 Timestamps

Converted to **microseconds**, then rebased so each trace starts at 0.

| Dataset | Native unit | Conversion |
| :--- | :--- | :--- |
| MSR | FILETIME (100 ns ticks) | `ticks // 10` |
| SYSTOR | fractional Unix seconds | `round(seconds * 1e6)` |
| FIU | nanoseconds | `ns // 1000` |

Microseconds were chosen because they are the finest unit all three sources
genuinely support (SYSTOR's own README disclaims its nanosecond digits), so no
source is given false precision. Rebasing to zero makes intervals comparable
across traces whose absolute epochs differ by decades.

Conversion happens **per record, before rebasing**, so the reported gap between
two records is the difference of their truncated microsecond values.

### 4.3 Offsets to block addresses, and multi-block expansion

A request covers **every block it touches**:

```
first_block = offset // block_size
last_block  = (offset + size - 1) // block_size
```

and one record is emitted per block in that range, all carrying the original
timestamp. At a 4 KB block size:

| Request | Blocks emitted |
| :--- | :--- |
| 4 KB, aligned | 1 |
| 20 KB, aligned | 5 |
| 128 KB, aligned | 32 |
| 8 KB starting 2 bytes into a block | **3** (partial first, whole middle, partial last) |

**Unaligned requests are not rounded away.** A request that begins part-way into
a block really does touch that block, so the block is emitted; the count of such
requests is reported as `unaligned_requests` so the frequency is visible rather
than assumed negligible.

Verified against genuine MSR records in `test_msr_parser_expands_and_converts`,
and again end-to-end: raw record `128166372011600556,mds,0,Write,3201662976,20480,45361`
produces exactly blocks 781656–781660 at timestamp 78997.

The simulator independently refuses any normalized record whose size is not
exactly one block, so an unexpanded request can never be silently accounted as
4 KB.

Requests longer than `preprocessing.max_blocks_per_request` (default 2048 blocks
= 8 MB) are treated as corrupt and counted, not expanded.

### 4.4 Operations

Mapped to `W` / `R` / `D`. An unrecognised value **keeps its spelling** and is
counted under `dropped_unknown_operation` with examples recorded — so an
unexpected format shows up as a named anomaly rather than as a quietly halved
trace. SYSTOR's blank `IOType` cannot be classified as a read or a write and is
dropped for the same reason, rather than guessed.

### 4.5 Device-namespace safety

**LBAs are only meaningful within one volume.** Disk 0 and disk 1 both have a
block 0; pooling them would invent rewrites between unrelated data and corrupt
every downstream measurement.

Each parser reports a device key per record — `<host>_<disk>` (MSR),
`major:minor` (FIU), `LUN` (SYSTOR) — and a normalized trace always describes
exactly one device. The default policy **rejects** a file containing more than
one, naming the devices found; `device="dominant"` keeps the busiest and records
the choice; an explicit key selects one.

This is why the normalized schema has no device column: within a file it would be
constant. The identity lives in `trace_id` and in `device_selected` in the
statistics JSON, so the provenance of a normalized file is never ambiguous.

SYSTOR shards one LUN across hourly files, so the pipeline groups consecutive
hours of the same LUN into a single chronological workload and then verifies that
no two LUNs were mixed.

### 4.6 Ordering and validation

* Records are **stable-sorted by timestamp**; any that arrived out of order are
  counted (`out_of_order_records`) rather than silently reordered. Ties keep
  their original file order.
* Duplicate timestamps are counted (`duplicate_timestamps`). They are normal —
  one expanded request produces several records at one instant.
* Rejected records are counted **by category**: malformed, bad offset, bad size,
  oversized request, unknown operation, other device. Nothing is discarded
  without a number attached.
* An empty or unreadable file produces an empty result with the counts intact,
  not a crash.

### 4.7 Truncation

`--max-records` caps how many raw records are consumed. The cap always keeps a
**chronological prefix**, never a sample, so causal ordering survives — and the
fact of truncation is recorded as `truncated_by_limit` in the statistics and in
`results/final/experiment_manifest.json`. Results describe that prefix, not the
whole week.

---

## 5. Rewrite intervals

For each LBA, the rewrite interval of a write is the time since the previous
write **to that same LBA**:

```
LBA 100 written at t = 10, 18, 24, 40   ->   intervals 8, 6, 16
```

* **The first write of an LBA has no interval and never becomes a training
  target.** A fabricated zero would teach the model that every newly written
  block is maximally hot.
* A genuine zero (two writes in the same microsecond) is a real observation and
  is kept.
* **Reads never produce rewrite intervals.** They may appear in a normalized
  trace and are ignored.

---

## 6. Workload selection

Candidate thresholds and the role-assignment rule are frozen in
`ml/dataset_selection.py` **before any model is trained**, and depend on no
measured model or WAF result. A workload must have at least 200,000 write blocks,
a rewrite ratio of at least 0.30, at least 5,000 distinct written LBAs and at
least an hour of activity. The full rule, and why it is mechanical, is in
[`docs/methodology.md`](methodology.md) §2.

Measured statistics for every candidate — request counts, write percentage,
unique LBAs, LBA range, total bytes, request sizes, duration, rewrite counts and
ratios, write-traffic concentration and rewrite-interval percentiles — are in
[`results/dataset_inventory.csv`](../results/dataset_inventory.csv), with the
per-workload detail in `results/dataset_statistics/*.json`.

---

## 7. Limitations

* **Age.** MSR Cambridge is 2007; SYSTOR '17 is 2016. SNIA files both under
  "Historical Traces". Neither represents contemporary storage hardware,
  interfaces or workloads.
* **Below the cache.** These are block-level traces captured beneath a file
  system and its page cache. They show what reached the device, not what the
  application asked for.
* **Truncated.** The experiments use chronological prefixes, not whole traces.
* **One device model.** Every workload is replayed against a single greedy-cleaned
  log-structured device with fixed block and segment sizes; no parallelism, wear
  levelling or read-latency modelling.
* **SYSTOR carries no publisher checksum**, so its integrity guarantee is weaker
  than MSR's — SHA-256 recorded at download time rather than a manifest written
  by the data's authors.
* **FIU is absent**, so the external-generalization claim rests on SYSTOR alone.
