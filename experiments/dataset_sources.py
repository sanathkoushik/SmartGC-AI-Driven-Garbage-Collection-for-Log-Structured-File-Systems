"""Recorded evaluation of candidate public block-I/O trace sources.

Every field below was checked directly on 2026-09-09 (HTTP probes, archive
index walks and README reads); none of it is recalled or assumed. The scoring
is mechanical so the choice can be re-derived rather than taken on trust, and
``experiments/analyze_datasets.py --sources`` regenerates
``results/dataset_source_comparison.csv`` from it.

Scores are 0-5.

``reproducibility_score``
    Can another researcher obtain the identical bytes from a documented URL
    with a scripted, unattended command?
        5  anonymous HTTPS, byte-range capable, publisher checksums available
        3  anonymous HTTPS, no checksums
        1  requires a human to fill a form or accept a licence in a browser
        0  host does not serve the files at all

``trust_score``
    Can provenance be established and attributed honestly?
        5  canonical repository, or an unmodified copy of the original
           distribution whose publisher is identifiable and whose contents match
           the canonical checksums
        3  reputable institution, contents not independently verifiable
        1  provenance unclear
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Any

VERIFIED_ON = "2026-09-09"


@dataclass(frozen=True)
class SourceRecord:
    dataset: str
    original_source: str
    host: str
    host_type: str
    paper: str
    license: str
    download_access: str
    format: str
    size: str
    trace_count: str
    request_fields: str
    write_support: str
    rewrite_analysis_possible: str
    reproducibility_score: int
    trust_score: int
    selected: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CSV_COLUMNS = [f.name for f in fields(SourceRecord)]


SOURCES: tuple[SourceRecord, ...] = (
    SourceRecord(
        dataset="MSR Cambridge (2007)",
        original_source="Microsoft Research Cambridge",
        host="cache-datasets S3 bucket (cacheMon / github.com/cacheMon/cache_dataset)",
        host_type="public S3 bucket published by a research group",
        paper=("Narayanan, Donnelly, Rowstron, 'Write Off-Loading: Practical Power "
               "Management for Enterprise Storage', USENIX FAST '08"),
        license="Microsoft trace DISCLAIMER.txt shipped inside the archive",
        download_access="anonymous HTTPS, no form, HTTP range requests supported",
        format="tar of per-volume gzipped CSV: Timestamp,Hostname,DiskNumber,Type,Offset,Size,ResponseTime",
        size="3.06 GB + 1.88 GB (two archives)",
        trace_count="36 volumes across 13 servers",
        request_fields="timestamp (Windows FILETIME), host, disk, R/W, byte offset, byte size, response time",
        write_support="yes, explicit Read/Write type",
        rewrite_analysis_possible="yes: byte offsets map to 4KB blocks and blocks are rewritten heavily",
        reproducibility_score=5,
        trust_score=5,
        selected="PRIMARY",
        notes=("Archives are the unmodified original distribution: they contain Microsoft's own "
               "README.txt, DISCLAIMER.txt and MD5.txt, and every downloaded volume was verified "
               "against those MD5 checksums. Individual volumes are extracted by byte range, so "
               "no multi-GB download is needed."),
    ),
    SourceRecord(
        dataset="MSR Cambridge (2007)",
        original_source="Microsoft Research Cambridge",
        host="SNIA IOTTA, iotta.snia.org/traces/block-io/388",
        host_type="canonical trace repository",
        paper="Narayanan, Donnelly, Rowstron, USENIX FAST '08",
        license="SNIA Trace Data Files Download License",
        download_access="BLOCKED for automation: web form requires name, e-mail and organisation",
        format="same archives as above",
        size="3.06 GB + 1.88 GB",
        trace_count="36 volumes",
        request_fields="as above",
        write_support="yes",
        rewrite_analysis_possible="yes",
        reproducibility_score=1,
        trust_score=5,
        selected="no (cited as the canonical repository)",
        notes=("The authoritative catalogue entry, and the citation SmartGC gives. Bulk download "
               "is gated behind a personal-details form, so it is not used as the fetch route. "
               "1000-record sample traces are downloadable without the form."),
    ),
    SourceRecord(
        dataset="SYSTOR '17 enterprise VDI (2016)",
        original_source="Fujitsu Laboratories Ltd.",
        host="cache-datasets S3 bucket (cacheMon)",
        host_type="public S3 bucket published by a research group",
        paper=("Lee, Kumano, Matsuki, Endo, Fukumoto, Sugawara, 'Understanding storage traffic "
               "characteristics on enterprise virtual desktop infrastructure', SYSTOR '17"),
        license="SNIA Trace Data Files Download License (distribution README included)",
        download_access="anonymous HTTPS, no form, HTTP range requests supported",
        format="tar of hourly gzipped CSV per LUN: Timestamp,Response,IOType,LUN,Offset,Size",
        size="~1.05 GB per tar; a byte prefix is fetched",
        trace_count="2706 hourly traces across 6 LUNs",
        request_fields="timestamp (fractional Unix seconds), response time, R/W, LUN, byte offset, byte size",
        write_support="yes, explicit R/W IOType",
        rewrite_analysis_possible="yes: byte offsets map to 4KB blocks, VDI workloads rewrite heavily",
        reproducibility_score=4,
        trust_score=4,
        selected="SECONDARY (external generalization)",
        notes=("Chosen as the external-validation workload after FIU proved unobtainable. A "
               "different workload family (2016 enterprise VDI) and nine years newer than MSR, "
               "so it is a stronger transfer test than another 2008 university trace. No "
               "publisher checksums are shipped, hence 4 rather than 5."),
    ),
    SourceRecord(
        dataset="FIU IODedup / SRCMap (2008-2009)",
        original_source="Florida International University, School of Computing and Information Sciences",
        host="SNIA IOTTA, iotta.snia.org/traces/block-io/390",
        host_type="canonical trace repository",
        paper=("Koller, Rangaswami, 'I/O Deduplication', FAST '10; Verma et al., 'SRCMap', FAST '10"),
        license="SNIA Trace Data Files Download License",
        download_access="BLOCKED: same personal-details form as MSR",
        format="whitespace-separated blkparse records",
        size="14.2 GB (IODedup) + 14.6 GB (SRCMap)",
        trace_count="10 IODedup subtraces, 9 SRCMap subtraces",
        request_fields="timestamp (ns), pid, process, LBA (512B sectors), size (512B blocks), R/W, major, minor, MD5",
        write_support="yes",
        rewrite_analysis_possible="yes",
        reproducibility_score=1,
        trust_score=5,
        selected="no (parser implemented, data not obtainable automatically)",
        notes=("Requested as the secondary dataset but not automatable. The parser is implemented "
               "and verified against real FIU sample records, so files dropped into data/raw/fiu/ "
               "are picked up with no code change."),
    ),
    SourceRecord(
        dataset="FIU IODedup / SRCMap (2008-2009)",
        original_source="Florida International University",
        host="Harvey Mudd mirror, iotta.cs.hmc.edu/traces/block-io/390",
        host_type="mirror of the canonical repository",
        paper="as above",
        license="SNIA Trace Data Files Download License",
        download_access="BLOCKED: runs the same gated application as SNIA (HTTP 307 to the form)",
        format="as above",
        size="as above",
        trace_count="as above",
        request_fields="as above",
        write_support="yes",
        rewrite_analysis_possible="yes",
        reproducibility_score=1,
        trust_score=4,
        selected="no",
        notes="Probed directly: /download?type=file redirects to /downloaderinfos/new, the same form.",
    ),
    SourceRecord(
        dataset="FIU block traces (2012)",
        original_source="ASU VISA Research Lab (Ming Zhao's group, formerly at FIU)",
        host="visa.lab.asu.edu/web/resources/traces/",
        host_type="research group web page",
        paper="Arteaga, Zhao, 'Client-side Flash Caching for Cloud Systems', SYSTOR '14",
        license="attribution requested on the page",
        download_access="BROKEN: every /traces/... link returns HTTP 404",
        format="tar.gz of blktrace output",
        size="unknown (files unavailable)",
        trace_count="5 servers listed (Webserver, Moodle, Buffalo, Bear, CloudVPS)",
        request_fields="unknown (files unavailable)",
        write_support="unknown",
        rewrite_analysis_possible="unknown",
        reproducibility_score=0,
        trust_score=3,
        selected="no",
        notes=("Pages still list the traces but the files are gone; the original FIU host "
               "sylab-srv.cs.fiu.edu does not respond either. Recorded so the dead end is not "
               "re-investigated."),
    ),
    SourceRecord(
        dataset="Alibaba Cloud EBS (2020)",
        original_source="Alibaba Group",
        host="cache-datasets S3 bucket (cacheMon)",
        host_type="public S3 bucket published by a research group",
        paper=("Li et al., 'An In-Depth Analysis of Cloud Block Storage Workloads in Large-Scale "
               "Production', IISWC '20"),
        license="Alibaba trace terms",
        download_access="anonymous HTTPS, but a single 150 GB zstd file",
        format="CSV: device_id, opcode (R/W), offset (bytes), length (bytes), timestamp (us)",
        size="150.4 GB compressed, single object",
        trace_count="1000 disks in one file",
        request_fields="timestamp, device, R/W, byte offset, byte length",
        write_support="yes",
        rewrite_analysis_possible="yes",
        reproducibility_score=2,
        trust_score=4,
        selected="no",
        notes=("Excellent fields and genuinely modern, but stored as one 150 GB zstd stream: a "
               "partial fetch cannot be decompressed from an arbitrary offset, so a usable subset "
               "cannot be obtained without downloading the whole object."),
    ),
    SourceRecord(
        dataset="Tencent Cloud EBS (2020)",
        original_source="Tencent",
        host="cache-datasets S3 bucket (cacheMon)",
        host_type="public S3 bucket published by a research group",
        paper="Zhang et al., 'OSCA', USENIX ATC '20",
        license="Tencent trace terms",
        download_access="anonymous HTTPS, single 137 GB zstd file",
        format="Timestamp, Offset (sectors), Size (sectors), IOType (0=read, 1=write), VolumeID",
        size="137.2 GB compressed, single object",
        trace_count="5584 volumes in one file",
        request_fields="timestamp, offset, size, R/W, volume",
        write_support="yes",
        rewrite_analysis_possible="yes",
        reproducibility_score=2,
        trust_score=4,
        selected="no",
        notes="Same single-large-object problem as the Alibaba traces.",
    ),
    SourceRecord(
        dataset="Twitter Twemcache / Meta KV / Wikimedia CDN",
        original_source="Twitter / Meta / Wikimedia",
        host="cache-datasets S3 bucket (cacheMon)",
        host_type="public S3 bucket published by a research group",
        paper="Yang et al., OSDI '20 and others",
        license="various",
        download_access="anonymous HTTPS",
        format="key-value and object-cache request traces",
        size="TB scale",
        trace_count="many",
        request_fields="timestamp, key, key size, value size, operation",
        write_support="GET/SET on keys, not block writes",
        rewrite_analysis_possible=("NO: these are object/key caches with no logical block address "
                                  "space, no byte offsets and no fixed-size blocks"),
        reproducibility_score=5,
        trust_score=5,
        selected="no (wrong data model)",
        notes=("Explicitly rejected. Treating a cache key as an LBA would fabricate a block "
               "address space that does not exist, and there is no notion of overwriting a 4KB "
               "block to garbage collect."),
    ),
)


def selected_sources() -> tuple[SourceRecord, ...]:
    return tuple(record for record in SOURCES if record.selected.startswith(("PRIMARY", "SECONDARY")))
