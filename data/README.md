# SmartGC data directory

```
data/
├── raw/                     downloaded traces (gitignored - see "What is not committed")
│   ├── msr/                 MSR Cambridge volumes + the distribution's README/MD5/DISCLAIMER
│   ├── systor/              SYSTOR '17 hourly LUN files + the distribution README
│   ├── fiu/                 empty unless you supply FIU files yourself
│   └── download_manifest.json   what was fetched, from where, with SHA-256 and byte range
├── processed/
│   ├── normalized/          one CSV per workload in the normalized contract
│   └── sequences/           cached (window -> next interval) arrays, .npz
└── predictions/             per-write model output for the simulator
```

## How to obtain the data

```bash
python scripts/download_datasets.py --dataset all
python scripts/verify_datasets.py
```

No account, no form, no credentials. Individual volumes are pulled out of the
published archives by HTTP range request, so the ~5 GB of archives are never
downloaded in full.

`--list` shows every available volume and its size without downloading anything.

## Provenance

**The host these bytes are fetched from is not the organisation that produced
them.** The full record, including the routes that were tried and rejected, is in
[`docs/dataset_source_decision.md`](../docs/dataset_source_decision.md).

### MSR Cambridge — primary dataset

| | |
| :--- | :--- |
| Original data | Microsoft Research Cambridge |
| Collected | 2007 |
| Scale | 36 volumes across 13 servers, one week (per the distribution README) |
| Citation | Narayanan, Donnelly, Rowstron, "Write Off-Loading: Practical Power Management for Enterprise Storage", USENIX FAST '08 |
| Canonical repository | <https://iotta.snia.org/traces/block-io/388> |
| Downloaded from | `https://cache-datasets.s3.amazonaws.com/cache_dataset_txt/2008_msr/msr-cambridge{1,2}.tar` |
| Terms | `DISCLAIMER.txt`, shipped inside the archive by Microsoft |
| Integrity | verified against `MD5.txt`, **written by the data's authors** and shipped in the same archive |

Record format, quoted from the distribution's `README.txt`:

```
Timestamp,Hostname,DiskNumber,Type,Offset,Size,ResponseTime

Timestamp is the time the I/O was issued in "Windows filetime"
Hostname is the hostname (should be the same as that in the trace file name)
DiskNumber is the disknumber (should be the same as in the trace file name)
Type is "Read" or "Write"
Offset is the starting offset of the I/O in bytes from the start of the logical disk.
Size is the transfer size of the I/O request in bytes.
ResponseTime is the time taken by the I/O to complete, in Windows filetime units.
```

Files are named `<hostname>_<disknumber>.csv.gz`, one logical volume each.

### SYSTOR '17 — external generalization dataset

| | |
| :--- | :--- |
| Original data | Fujitsu Laboratories Ltd. |
| Collected | February–March 2016 |
| Scale | 2,706 hourly traces across 6 LUNs (per the distribution README) |
| Citation | Lee, Kumano, Matsuki, Endo, Fukumoto, Sugawara, "Understanding storage traffic characteristics on enterprise virtual desktop infrastructure", SYSTOR '17 |
| Canonical repository | <https://iotta.snia.org/traces/block-io/4928> |
| Downloaded from | `https://cache-datasets.s3.amazonaws.com/cache_dataset_txt/2017_systor/tar/systor-17-01.tar` |
| Terms | SNIA Trace Data Files Download License |
| Integrity | SHA-256 recorded in `download_manifest.json` at download time (no publisher checksum ships with this distribution) |

Record format, quoted from the distribution's `README.txt`:

```
Timestamp,Response,IOType,LUN,Offset,Size

Timestamp is the time the I/O was issued, as Unix time with a fractional part
  (nine digits, but accurate only to the microsecond).
Response is the time needed to complete the I/O.
IOType is "Read(R)", "Write(W)", or ""(blank).
```

Files are named `<YYYYMMDDHH>-LUN<n>.csv.gz` — one hour, one LUN. The pipeline
groups consecutive hours of the same LUN into one chronological workload.

Only a byte prefix of the first archive is downloaded. Members are stored
contiguously, so a prefix yields whole hourly files for every LUN; the member cut
short at the end of the prefix is discarded.

### FIU — requested, not obtainable

`data/raw/fiu/` exists but is empty. FIU traces cannot be downloaded without
submitting a name, e-mail address and organisation to the SNIA IOTTA form, the
Harvey Mudd mirror runs the same gated application, the original FIU host is
offline, and every ASU VISA Lab download link returns HTTP 404. All four were
probed directly on 2026-09-09.

The FIU parser is implemented and tested. If you obtain the files, drop any
plain/`.gz`/`.bz2`/`.xz` blkparse-format file into `data/raw/fiu/` and run:

```bash
python scripts/verify_datasets.py
python -m ml.preprocessing.normalize --discover
```

They will be picked up with no code change. SmartGC does not depend on them:
SYSTOR '17 serves as the external-generalization workload.

## What is not committed

**No trace data is in this repository.** Microsoft's `DISCLAIMER.txt` reserves
reproduction rights for the MSR files, and the normalized and sequence
intermediates run to gigabytes. Everything under `data/raw/`, `data/processed/`
and `data/predictions/` is gitignored and regenerated from the documented URLs.

What *is* committed is the means to reproduce it: the fetch script, the verifier,
the manifest schema, and every measured statistic in `results/`.

Unit tests therefore run on small synthetic files written in the real formats.
The tests that need genuine data (`test_real_msr_trace_parses`) skip themselves
unless the data has been downloaded.

## Integrity checking

```bash
python scripts/verify_datasets.py            # report
python scripts/verify_datasets.py --strict   # non-zero exit if anything is unverified
```

Three checks, strongest first:

1. **Publisher checksums** — MSR volumes against Microsoft's `MD5.txt`. This
   proves the bytes are the ones the authors released, not merely the ones the
   mirror served.
2. **Download manifest** — SHA-256 recorded when each file was fetched, catching
   post-download corruption where no publisher checksum exists.
3. **Structural check** — the first 2,000 records of every file are parsed with
   that dataset's own parser, so a truncated or wrong-format file is caught before
   it can become a spurious "workload property" downstream.

A file with no checksum from either source is reported as `UNVERIFIED`, never as
passing.
