# Dataset source decision

**Decided:** 2026-09-09. **Verified by direct HTTP probes on that date**, not from
memory or documentation.

This record exists so the choice of data source can be re-derived rather than
taken on trust. The scored comparison of every candidate is in
`results/dataset_source_comparison.csv`, generated from `experiments/dataset_sources.py`
by `python experiments/analyze_datasets.py --sources`.

---

## Decision

| Role | Dataset | Downloaded from |
| :--- | :--- | :--- |
| **Primary** | MSR Cambridge block I/O traces (2007) | `cache-datasets` public S3 bucket |
| **Secondary / external generalization** | SYSTOR '17 enterprise VDI traces (2016) | the same bucket |
| **Not used** | FIU IODedup / SRCMap | not obtainable without a personal-details form |

Both selected datasets download **anonymously over HTTPS with no form, no
account and no credentials**, and both support HTTP range requests so individual
volumes are extracted without pulling whole multi-gigabyte archives.

---

## Provenance, stated precisely

The distinction between *who produced the data* and *where we fetched the bytes*
matters and is kept explicit everywhere in this repository.

### MSR Cambridge

* **Original data producer:** Microsoft Research Cambridge.
* **Original publication (required citation):** Dushyanth Narayanan, Austin
  Donnelly, Antony Rowstron, "Write Off-Loading: Practical Power Management for
  Enterprise Storage", *6th USENIX Conference on File and Storage Technologies
  (FAST '08)*.
* **Canonical repository:** SNIA IOTTA, <https://iotta.snia.org/traces/block-io/388>
  (subtraces 386 and 387).
* **Downloaded from:** `https://cache-datasets.s3.amazonaws.com/cache_dataset_txt/2008_msr/msr-cambridge1.tar`
  and `msr-cambridge2.tar` — a public S3 bucket published by the cacheMon group
  (<https://github.com/cacheMon/cache_dataset>).
* **Licence/terms:** the `DISCLAIMER.txt` Microsoft ships inside the archive.
  It reserves reproduction rights, so **no MSR trace bytes are committed to this
  repository**; only the fetch script is.

**Why this host is trustworthy for these bytes.** The bucket does not hold a
re-processed derivative. Walking the tar index shows the original distribution
intact:

```
MSR-Cambridge/DISCLAIMER.txt        1,262 bytes
MSR-Cambridge/MD5.txt               1,712 bytes
MSR-Cambridge/README.txt            1,815 bytes
MSR-Cambridge/hm_0.csv.gz      41,967,571 bytes
... 36 volume files in total across the two archives
```

`MD5.txt` is Microsoft's own checksum manifest, listing all 36 volumes. **Every
volume this project downloads is verified against it**, and a mismatch deletes
the file and aborts. That is a stronger provenance guarantee than the canonical
repository itself offers over HTTP, because it is a checksum written by the data's
authors rather than a promise made by a host. The `README.txt` retrieved from the
bucket is byte-for-byte the same 1,815-byte file SNIA serves.

### SYSTOR '17

* **Original data producer:** Fujitsu Laboratories Ltd.
* **Original publication:** Chunghan Lee, Tatsuo Kumano, Tatsuma Matsuki,
  Hiroshi Endo, Naoto Fukumoto, Mariko Sugawara, "Understanding storage traffic
  characteristics on enterprise virtual desktop infrastructure", *SYSTOR '17*.
* **Canonical repository:** SNIA IOTTA, <https://iotta.snia.org/traces/block-io/4928>.
* **Downloaded from:** `https://cache-datasets.s3.amazonaws.com/cache_dataset_txt/2017_systor/tar/systor-17-01.tar`.
* **Licence/terms:** SNIA Trace Data Files Download License; the distribution
  `README.txt` is fetched alongside the data.

No publisher checksum file ships with this distribution, so SYSTOR files are
verified against the SHA-256 recorded in `data/raw/download_manifest.json` at
download time rather than against an authors' manifest. That is a weaker
guarantee and is scored as such (reproducibility 4 rather than 5).

---

## Candidates investigated

### 1. SNIA IOTTA — MSR Cambridge (canonical) — **rejected as a fetch route**

*Strengths:* the authoritative catalogue; the citation this project gives.

*Weakness that decided it:* `GET /traces/block-io/386/download?type=file` returns
HTTP 307 to `/downloaderinfos/new`, a form whose required fields are first name,
last name, e-mail address and organisation, ending in a "Yes, I Accept" licence
button. Automating that would mean submitting the user's personal details and
accepting a licence on their behalf. Not done.

*Note:* `download?type=sample_trace` and `type=readme` are **not** gated. Real
1,000-record samples were retrieved this way and used to confirm the record
formats before any parser was written.

### 2. cacheMon `cache-datasets` S3 bucket — **selected**

*Strengths:* anonymous HTTPS; bucket listing is public; `Accept-Ranges: bytes`;
holds the **unmodified original archives** including the authors' `MD5.txt`;
published by an identifiable research group alongside a documented GitHub
repository; also carries SYSTOR '17, so one host covers both datasets.

*Weaknesses:* a third-party mirror, so it is described as a download location and
never as the data's origin. Mitigated by verifying against Microsoft's checksums,
which detects any alteration.

*Also present but rejected:* Alibaba Cloud EBS 2020 (150 GB) and Tencent Cloud
EBS 2020 (137 GB) are each stored as a **single zstd object**. A byte-range
prefix of a single-frame zstd stream cannot be decompressed independently, so
there is no way to obtain a usable subset short of the whole object. Excellent
data, impractical packaging. The bucket's Twitter / Meta / Wikimedia sets are
**key-value and CDN object traces with no logical block address space, no byte
offsets and no fixed-size blocks** — treating a cache key as an LBA would
fabricate an address space that does not exist, so they were excluded on data-model
grounds, not on convenience.

### 3. FIU IODedup / SRCMap — **wanted, but unobtainable**

Three independent routes were probed and all failed:

| Route | Result |
| :--- | :--- |
| SNIA IOTTA `/traces/block-io/390` | HTTP 307 to the personal-details form |
| Harvey Mudd mirror `iotta.cs.hmc.edu/traces/block-io/390` | the same gated application |
| Original FIU host `sylab-srv.cs.fiu.edu` | connection times out |
| ASU VISA Lab `visa.lab.asu.edu/web/resources/traces/` | pages still list FIU block traces, every `/traces/...` link returns **HTTP 404** |

The FIU parser is implemented, documented and unit-tested (including against the
real FIU sample records, which *are* downloadable without the form). Any FIU
files placed in `data/raw/fiu/` are picked up by `--discover` with no code change.
`python scripts/download_datasets.py --dataset fiu` prints the manual procedure
and the evidence above.

### 4. Why SYSTOR '17 replaced FIU as the external validation set

FIU would have been a 2008–2009 university-server workload — the same era and
broadly the same setting as MSR. SYSTOR '17 is **2016 enterprise virtual desktop
infrastructure**: nine years newer and a different workload family. For the
question this project asks (does a model pretrained on one workload family
transfer to another?) it is the stronger test, not merely the available one.

---

## Suitability check

Both selected datasets carry everything the study needs, verified by parsing real
records rather than by reading documentation:

| Requirement | MSR Cambridge | SYSTOR '17 |
| :--- | :--- | :--- |
| Timestamp | Windows FILETIME, 100 ns ticks | fractional Unix seconds, µs-accurate |
| Logical location | byte offset from volume start | byte offset from LUN start |
| Request size | bytes | bytes |
| Read/write distinguished | `Type` = Read/Write | `IOType` = R/W (blank rows dropped) |
| Device identity | `Hostname` + `DiskNumber` | `LUN` |
| Repeated writes to the same block | yes — measured rewrite ratio 0.936 on `mds_0` | measured per LUN in the inventory |

The last row is the one that actually matters: a dataset without repeated writes
to the same logical block has no rewrite intervals to predict and the research
question would be undefined on it. `results/dataset_inventory.csv` reports the
measured rewrite ratio for every workload, and `ml/dataset_selection.py` excludes
any workload below 0.30.

---

## Honest caveats

* **MSR Cambridge is from 2007 and SYSTOR '17 from 2016.** SNIA files both under
  "Historical Traces". They remain the standard public block-I/O corpora for this
  kind of study, but nothing here claims they represent contemporary storage.
* **The download host is not the data's producer**, and this repository never
  says otherwise.
* **No MSR trace data is committed**, per Microsoft's disclaimer. The pipeline
  reproduces the data from the documented URLs instead.
* The CMU PDL mirror sometimes cited for MSR traces **does not host them** — its
  `/pub/datasets/` index contains ATLAS, Baleen24, cacheDatasets, hla, ottertune
  and twemcacheWorkload only. It is not cited here.
