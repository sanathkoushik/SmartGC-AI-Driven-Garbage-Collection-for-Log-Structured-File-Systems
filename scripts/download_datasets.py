#!/usr/bin/env python3
"""Download the real block-I/O traces SmartGC is evaluated on.

    python scripts/download_datasets.py --dataset msr
    python scripts/download_datasets.py --dataset systor
    python scripts/download_datasets.py --dataset fiu        # prints manual steps
    python scripts/download_datasets.py --list               # show what is available

Provenance
----------
MSR Cambridge and SYSTOR '17 are **not** produced by the host used here. See
docs/dataset_source_decision.md for the full record; in short:

  MSR Cambridge
    Original data  : Microsoft Research Cambridge (Narayanan, Donnelly,
                     Rowstron; USENIX FAST '08)
    Canonical repo : SNIA IOTTA, https://iotta.snia.org/traces/block-io/388
    Downloaded from: the public `cache-datasets` S3 bucket published by the
                     cacheMon group (https://github.com/cacheMon/cache_dataset),
                     which stores the ORIGINAL distribution tarballs unmodified.
    Verified by    : the MD5.txt shipped inside those tarballs by Microsoft.

  SYSTOR '17
    Original data  : Fujitsu Laboratories (Lee, Kumano, Matsuki, Endo,
                     Fukumoto, Sugawara; SYSTOR '17)
    Canonical repo : SNIA IOTTA, https://iotta.snia.org/traces/block-io/4928
    Downloaded from: the same public `cache-datasets` S3 bucket.

Both archives are served anonymously over HTTPS and support HTTP range
requests, so individual volumes are fetched without downloading whole
multi-gigabyte archives.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"

S3_BASE = "https://cache-datasets.s3.amazonaws.com"
MSR_ARCHIVES = [
    f"{S3_BASE}/cache_dataset_txt/2008_msr/msr-cambridge1.tar",
    f"{S3_BASE}/cache_dataset_txt/2008_msr/msr-cambridge2.tar",
]
SYSTOR_ARCHIVE = f"{S3_BASE}/cache_dataset_txt/2017_systor/tar/systor-17-01.tar"
SYSTOR_README = f"{S3_BASE}/cache_dataset_txt/2017_systor/README.txt"

# Default MSR volumes. Chosen before any model was trained, by the criteria in
# docs/methodology.md: moderate compressed size (tractable on one workstation)
# and coverage of distinct server roles. The held-out target and the FIU-style
# external set are named there too, not here, so this list stays a pure
# acquisition concern.
DEFAULT_MSR_VOLUMES = [
    "hm_0", "mds_0", "prn_0", "proj_0", "rsrch_0", "src2_0",
    "stg_0", "ts_0", "usr_0", "wdev_0", "web_0", "prxy_0",
]

# One tar prefix is enough for the SYSTOR subset: members are stored
# contiguously, so a byte prefix yields whole hourly files for every LUN.
SYSTOR_DEFAULT_PREFIX_MB = 160

CHUNK = 1 << 20


@dataclass
class DownloadRecord:
    dataset: str
    name: str
    path: str
    bytes: int
    sha256: str
    expected_md5: str | None
    md5_verified: bool | None
    source_url: str
    byte_range: str | None


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} GB"


def http_range(session: requests.Session, url: str, start: int, length: int) -> bytes:
    """Fetch exactly `length` bytes starting at `start`."""
    end = start + length - 1
    response = session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
    if response.status_code not in (200, 206):
        raise RuntimeError(f"HTTP {response.status_code} for range {start}-{end} of {url}")
    return response.content


def stream_range_to_file(session: requests.Session,
                         url: str,
                         start: int,
                         length: int,
                         destination: Path,
                         label: str,
                         attempts: int = 6) -> tuple[int, str]:
    """Stream a byte range to `destination`, returning (bytes written, sha256).

    Written to a `.part` file first, so an interrupted transfer can never be
    mistaken for a complete file. A broken connection resumes from whatever the
    `.part` already holds rather than starting over: these archives are
    gigabytes and a mid-stream drop is normal on a long transfer.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    # A stale .part longer than the target cannot be a prefix of it.
    if partial.exists() and partial.stat().st_size > length:
        partial.unlink()

    started = time.monotonic()
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        written = partial.stat().st_size if partial.exists() else 0
        if written >= length:
            break
        try:
            headers = {"Range": f"bytes={start + written}-{start + length - 1}"}
            with session.get(url, headers=headers, stream=True,
                             timeout=(30, 120)) as response:
                if response.status_code not in (200, 206):
                    raise RuntimeError(
                        f"HTTP {response.status_code} downloading {label} from {url}")
                if written and response.status_code == 200:
                    # The server ignored the Range header, so the body starts at
                    # byte 0 and cannot be appended to what we already have.
                    partial.unlink(missing_ok=True)
                    written = 0
                with partial.open("ab" if written else "wb") as handle:
                    for block in response.iter_content(chunk_size=CHUNK):
                        if not block:
                            continue
                        if written + len(block) > length:
                            block = block[: length - written]
                        handle.write(block)
                        written += len(block)
                        elapsed = max(time.monotonic() - started, 1e-6)
                        print(f"\r  {label}: {100.0 * written / length:5.1f}%"
                              f"  {_human(written)} / {_human(length)}"
                              f"  ({_human(written / elapsed)}/s)   ",
                              end="", flush=True)
                        if written >= length:
                            break
            if written >= length:
                break
        except (requests.RequestException, OSError) as error:
            last_error = error
            delay = min(2 ** attempt, 30)
            print(f"\n  {label}: transfer interrupted ({type(error).__name__}); "
                  f"resuming from {_human(written)} in {delay}s "
                  f"(attempt {attempt}/{attempts})", flush=True)
            time.sleep(delay)
    print()

    final_size = partial.stat().st_size if partial.exists() else 0
    if final_size != length:
        raise RuntimeError(
            f"{label}: expected {length} bytes, have {final_size} after "
            f"{attempts} attempt(s)"
            + (f"; last error: {last_error}" if last_error else "")
        )

    # Hash the finished file: the content may have arrived across several
    # resumed connections, so an incremental digest would not be reliable.
    digest = hashlib.sha256(partial.read_bytes()).hexdigest()
    partial.replace(destination)
    return final_size, digest


def walk_tar_index(session: requests.Session, url: str, total_size: int,
                   max_members: int = 200) -> list[tuple[str, int, int]]:
    """Enumerate (name, size, data_offset) of a remote tar via range requests.

    Only the 512-byte headers are fetched, so indexing a 3 GB archive costs a
    few kilobytes.
    """
    members: list[tuple[str, int, int]] = []
    offset = 0
    for _ in range(max_members):
        header = http_range(session, url, offset, 512)
        if len(header) < 512:
            break
        raw_name = header[0:100].rstrip(b"\x00")
        if not raw_name.strip():
            break
        name = raw_name.decode("utf-8", "replace")
        try:
            size = int(header[124:135].rstrip(b"\x00 ").decode() or "0", 8)
        except ValueError:
            break
        members.append((name, size, offset + 512))
        offset += 512 + ((size + 511) // 512) * 512
        if offset >= total_size:
            break
    return members


def content_length(session: requests.Session, url: str) -> int:
    response = session.head(url, timeout=60, allow_redirects=True)
    response.raise_for_status()
    length = response.headers.get("Content-Length")
    if length is None:
        raise RuntimeError(f"No Content-Length for {url}")
    return int(length)


def parse_md5_file(text: str) -> dict[str, str]:
    """Parse the `MD5.txt` shipped inside the MSR distribution."""
    checksums: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        digest, name = parts[0], parts[-1].lstrip("*")
        checksums[name] = digest.lower()
    return checksums


def download_msr(session: requests.Session,
                 volumes: Iterable[str],
                 output_dir: Path,
                 force: bool = False) -> list[DownloadRecord]:
    wanted = {f"{name}.csv.gz" for name in volumes}
    records: list[DownloadRecord] = []
    remaining = set(wanted)

    for archive in MSR_ARCHIVES:
        if not remaining:
            break
        print(f"\nIndexing {archive}")
        total = content_length(session, archive)
        index = walk_tar_index(session, archive, total, max_members=80)
        print(f"  {len(index)} members, {_human(total)} archive")

        by_base = {Path(name).name: (name, size, offset) for name, size, offset in index}

        checksums: dict[str, str] = {}
        if "MD5.txt" in by_base:
            _, size, offset = by_base["MD5.txt"]
            checksums = parse_md5_file(http_range(session, archive, offset, size)
                                       .decode("utf-8", "replace"))
            print(f"  official MD5.txt lists {len(checksums)} files")

        # Keep the distribution's own documentation next to the data.
        for doc in ("README.txt", "DISCLAIMER.txt", "MD5.txt"):
            if doc in by_base:
                _, size, offset = by_base[doc]
                target = output_dir / doc
                if force or not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(http_range(session, archive, offset, size))

        for base in sorted(remaining & set(by_base)):
            _, size, offset = by_base[base]
            destination = output_dir / base
            expected = checksums.get(base)

            if destination.exists() and not force:
                print(f"  {base}: already present ({_human(destination.stat().st_size)}), skipping")
                sha = hashlib.sha256(destination.read_bytes()).hexdigest()
                md5_ok = None
                if expected:
                    md5_ok = hashlib.md5(destination.read_bytes()).hexdigest() == expected
                records.append(DownloadRecord("msr", base, str(destination.relative_to(REPO_ROOT)),
                                              destination.stat().st_size, sha, expected, md5_ok,
                                              archive, f"{offset}-{offset + size - 1}"))
                remaining.discard(base)
                continue

            try:
                written, sha = stream_range_to_file(session, archive, offset, size,
                                                    destination, base)
            except RuntimeError as error:
                # Report and move on: one unreachable volume should not cost the
                # whole run. The manifest records only what actually arrived.
                print(f"  {base}: FAILED - {error}", file=sys.stderr)
                continue
            md5_ok: bool | None = None
            if expected:
                md5_ok = hashlib.md5(destination.read_bytes()).hexdigest() == expected
                print(f"  MD5 {'OK' if md5_ok else 'MISMATCH'} ({expected})")
                if not md5_ok:
                    destination.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"{base}: MD5 does not match the checksum shipped by the data's authors. "
                        "The download was discarded."
                    )
            records.append(DownloadRecord("msr", base, str(destination.relative_to(REPO_ROOT)),
                                          written, sha, expected, md5_ok, archive,
                                          f"{offset}-{offset + size - 1}"))
            remaining.discard(base)

    if remaining:
        print(f"\nWARNING: not found in either archive: {sorted(remaining)}", file=sys.stderr)
    return records


def download_systor(session: requests.Session,
                    output_dir: Path,
                    prefix_mb: int = SYSTOR_DEFAULT_PREFIX_MB,
                    luns: Iterable[str] | None = None,
                    force: bool = False) -> list[DownloadRecord]:
    """Fetch a contiguous prefix of the first SYSTOR tar and unpack its members.

    Members are stored back to back, so a byte prefix yields whole hourly files.
    This avoids the ~1 GB full-archive download while still producing several
    consecutive hours for every LUN.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    readme = output_dir / "README.txt"
    if force or not readme.exists():
        response = session.get(SYSTOR_README, timeout=120)
        response.raise_for_status()
        readme.write_bytes(response.content)

    total = content_length(session, SYSTOR_ARCHIVE)
    length = min(prefix_mb * 1024 * 1024, total)
    print(f"\nFetching first {_human(length)} of {SYSTOR_ARCHIVE} ({_human(total)} total)")

    buffer = io.BytesIO()
    digest_all = hashlib.sha256()
    written = 0
    started = time.monotonic()
    with session.get(SYSTOR_ARCHIVE, headers={"Range": f"bytes=0-{length - 1}"},
                     stream=True, timeout=600) as response:
        if response.status_code not in (200, 206):
            raise RuntimeError(f"HTTP {response.status_code} downloading SYSTOR archive prefix")
        for block in response.iter_content(chunk_size=CHUNK):
            if not block:
                continue
            buffer.write(block)
            digest_all.update(block)
            written += len(block)
            elapsed = max(time.monotonic() - started, 1e-6)
            print(f"\r  systor-17-01.tar prefix: {100.0 * written / length:5.1f}%"
                  f"  {_human(written)} / {_human(length)}  ({_human(written / elapsed)}/s)   ",
                  end="", flush=True)
    print()

    buffer.seek(0)
    records: list[DownloadRecord] = []
    wanted = {lun.upper() for lun in luns} if luns else None

    # The prefix ends mid-member; stream reading stops cleanly at that point.
    with tarfile.open(fileobj=buffer, mode="r|") as archive:
        try:
            for member in archive:
                if not member.isfile():
                    continue
                base = Path(member.name).name
                if base.startswith("._") or base == "README.txt":
                    continue
                if wanted and not any(f"-{lun}." in base for lun in wanted):
                    continue
                destination = output_dir / base
                if destination.exists() and not force:
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                payload = extracted.read()
                if len(payload) != member.size:
                    break                      # truncated final member
                destination.write_bytes(payload)
                records.append(DownloadRecord(
                    "systor", base, str(destination.relative_to(REPO_ROOT)), len(payload),
                    hashlib.sha256(payload).hexdigest(), None, None,
                    SYSTOR_ARCHIVE, f"0-{length - 1}"))
        except (tarfile.ReadError, EOFError):
            pass                               # expected: the prefix cuts a member short

    print(f"  extracted {len(records)} hourly LUN files into {output_dir}")
    return records


FIU_MANUAL_INSTRUCTIONS = """
FIU traces cannot be downloaded automatically.

  Verified on {date}:
    * SNIA IOTTA (https://iotta.snia.org/traces/block-io/390) serves the bulk
      files only after a web form that requires a name, e-mail address and
      organisation, plus acceptance of the SNIA download licence. Submitting
      that form on your behalf is not something this script will do.
    * The Harvey Mudd mirror (http://iotta.cs.hmc.edu/traces/block-io/390) runs
      the same gated application.
    * The original FIU host (sylab-srv.cs.fiu.edu) does not respond.
    * The ASU VISA Lab trace pages (https://visa.lab.asu.edu/web/resources/traces/)
      still list FIU block traces, but every download link under /traces/
      returns HTTP 404.

If you want FIU included, obtain it yourself and drop the files in:

    data/raw/fiu/

Any plain, .gz, .bz2 or .xz file of blkparse-style records is accepted:

    [ts in ns] [pid] [process] [lba] [size in 512B blocks] [W|R] [major] [minor] [md5]

Then run:

    python scripts/verify_datasets.py
    python -m ml.preprocessing.normalize --discover

The FIU parser (ml/preprocessing/fiu_parser.py) is implemented and tested
against real FIU sample records, so the pipeline will pick the files up with no
further changes.

SmartGC does not depend on FIU: SYSTOR '17 (2016 enterprise VDI, Fujitsu) is
used as the external-generalization workload instead, and it downloads
automatically.
""".strip()


def write_manifest(records: list[DownloadRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict] = {}
    if path.exists():
        try:
            existing = {entry["path"]: entry for entry in json.loads(path.read_text())["files"]}
        except (json.JSONDecodeError, KeyError, TypeError):
            existing = {}
    for record in records:
        existing[record.path] = asdict(record)
    payload = {
        "generated_by": "scripts/download_datasets.py",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": sorted(existing.values(), key=lambda entry: (entry["dataset"], entry["name"])),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nManifest written: {path.relative_to(REPO_ROOT)}  ({len(payload['files'])} files)")


def list_available(session: requests.Session) -> int:
    print("MSR Cambridge volumes available in the public archives:\n")
    for archive in MSR_ARCHIVES:
        total = content_length(session, archive)
        print(f"  {archive}  ({_human(total)})")
        for name, size, _ in walk_tar_index(session, archive, total, max_members=80):
            base = Path(name).name
            if base.endswith(".csv.gz"):
                print(f"      {base:<18} {_human(size)}")
    print(f"\nSYSTOR '17 archive: {SYSTOR_ARCHIVE}")
    print(f"  ({_human(content_length(session, SYSTOR_ARCHIVE))}; a prefix is downloaded)")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["msr", "systor", "fiu", "all"])
    parser.add_argument("--volumes", nargs="*", default=None,
                        help="MSR volume names without extension, e.g. hm_0 prxy_0")
    parser.add_argument("--all-volumes", action="store_true",
                        help="MSR: download every volume in both archives (~5 GB)")
    parser.add_argument("--systor-prefix-mb", type=int, default=SYSTOR_DEFAULT_PREFIX_MB)
    parser.add_argument("--systor-luns", nargs="*", default=None,
                        help="e.g. LUN0 LUN1; default keeps every LUN in the prefix")
    parser.add_argument("--output-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument("--list", action="store_true", help="list remote contents and exit")
    args = parser.parse_args(list(argv) if argv is not None else None)

    session = requests.Session()
    session.headers["User-Agent"] = "SmartGC-dataset-fetcher/1.0 (research; +github.com/tarunts10)"

    if args.list:
        return list_available(session)
    if not args.dataset:
        parser.error("--dataset is required (or use --list)")

    records: list[DownloadRecord] = []
    manifest_path = args.output_dir / "download_manifest.json"

    try:
        if args.dataset in ("msr", "all"):
            volumes = args.volumes if args.volumes else DEFAULT_MSR_VOLUMES
            if args.all_volumes:
                volumes = []
                for archive in MSR_ARCHIVES:
                    total = content_length(session, archive)
                    volumes += [Path(n).name[:-len(".csv.gz")]
                                for n, _, _ in walk_tar_index(session, archive, total, 80)
                                if n.endswith(".csv.gz")]
                volumes = sorted(set(volumes))
            print(f"MSR volumes requested: {', '.join(volumes)}")
            records += download_msr(session, volumes, args.output_dir / "msr", force=args.force)

        if args.dataset in ("systor", "all"):
            records += download_systor(session, args.output_dir / "systor",
                                       prefix_mb=args.systor_prefix_mb,
                                       luns=args.systor_luns, force=args.force)

        if args.dataset in ("fiu", "all"):
            print("\n" + "=" * 72)
            print(FIU_MANUAL_INSTRUCTIONS.format(date="2026-09-09"))
            print("=" * 72)
            (args.output_dir / "fiu").mkdir(parents=True, exist_ok=True)
    except (requests.RequestException, RuntimeError) as error:
        print(f"\nDOWNLOAD FAILED: {error}", file=sys.stderr)
        if records:
            write_manifest(records, manifest_path)
        return 1

    if records:
        write_manifest(records, manifest_path)
        total_bytes = sum(record.bytes for record in records)
        verified = sum(1 for record in records if record.md5_verified)
        print(f"Downloaded {len(records)} file(s), {_human(total_bytes)}"
              f"; {verified} verified against the authors' MD5 checksums.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
