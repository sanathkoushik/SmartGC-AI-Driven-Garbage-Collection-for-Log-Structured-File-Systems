#!/usr/bin/env python3
"""Verify the raw traces in data/raw/ against their publishers' checksums.

    python scripts/verify_datasets.py
    python scripts/verify_datasets.py --strict     # non-zero exit on any problem

Checks, in order of strength:

1. **Publisher checksums.** The MSR Cambridge distribution ships ``MD5.txt``
   written by the data's authors. Any volume listed there is checked against it,
   which proves the bytes are the original ones and not something a mirror
   altered.
2. **Download manifest.** ``data/raw/download_manifest.json`` records the SHA-256
   and byte range of everything ``download_datasets.py`` fetched, so a file that
   changed after download is detected even where no publisher checksum exists.
3. **Structural sanity.** Every trace file is opened and its first records are
   parsed with the dataset's own parser, so a truncated or wrong-format file is
   caught before it silently becomes a "workload property" downstream.

Files present without any checksum are reported as UNVERIFIED rather than passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.preprocessing.normalize import discover_raw_traces              # noqa: E402
from ml.preprocessing.parsers import get_parser                         # noqa: E402

RAW_DIR = REPO_ROOT / "data" / "raw"
MANIFEST_PATH = RAW_DIR / "download_manifest.json"


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_md5_file(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    if not path.is_file():
        return checksums
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            checksums[parts[-1].lstrip("*")] = parts[0].lower()
    return checksums


def load_manifest() -> dict[str, dict[str, Any]]:
    if not MANIFEST_PATH.is_file():
        return {}
    try:
        payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        return {entry["path"]: entry for entry in payload.get("files", [])}
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}


def structural_check(dataset: str, paths: list[Path]) -> tuple[bool, str]:
    """Parse a small prefix so a truncated or mis-formatted file fails loudly."""
    try:
        parser = get_parser(dataset, device="dominant")
        table, stats = parser.parse(paths, trace_id="verify", max_records=2000)
    except Exception as error:                      # noqa: BLE001 - reported to the user
        return False, f"parse failed: {type(error).__name__}: {error}"

    if stats.raw_records_read == 0:
        return False, "no records could be read"
    if len(table) == 0:
        return False, f"{stats.raw_records_read} records read but none survived validation"
    dropped_fraction = stats.dropped_total / max(stats.raw_records_read, 1)
    if dropped_fraction > 0.5:
        return False, (f"{stats.dropped_total}/{stats.raw_records_read} records rejected "
                       "- wrong format for this dataset?")
    return True, (f"{stats.raw_records_read} records parsed, {len(table)} blocks, "
                  f"{stats.dropped_total} rejected")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero if anything is unverified or fails")
    parser.add_argument("--skip-structural", action="store_true",
                        help="checksum only; do not parse the files")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not RAW_DIR.is_dir():
        print(f"No raw data directory at {RAW_DIR}.")
        print("Run: python scripts/download_datasets.py --dataset all")
        return 2

    manifest = load_manifest()
    msr_checksums = parse_md5_file(RAW_DIR / "msr" / "MD5.txt")
    workloads = discover_raw_traces()

    if not workloads:
        print(f"No trace files found under {RAW_DIR}.")
        print("Run: python scripts/download_datasets.py --dataset all")
        return 2

    print(f"Verifying {len(workloads)} workload(s) under {RAW_DIR.relative_to(REPO_ROOT)}\n")
    verified = 0
    unverified = 0
    failed = 0

    for dataset, trace_id, paths in workloads:
        print(f"[{dataset}] {trace_id}")
        for path in paths:
            relative = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
            size = path.stat().st_size
            status = "UNVERIFIED"
            detail = "no publisher checksum and not in the download manifest"

            if dataset == "msr" and path.name in msr_checksums:
                expected = msr_checksums[path.name]
                actual = file_digest(path, "md5")
                if actual == expected:
                    status, detail = "OK", f"MD5 matches the authors' MD5.txt ({expected})"
                    verified += 1
                else:
                    status = "FAILED"
                    detail = f"MD5 mismatch: expected {expected}, got {actual}"
                    failed += 1
            elif relative in manifest:
                entry = manifest[relative]
                actual = file_digest(path, "sha256")
                if actual == entry.get("sha256"):
                    status, detail = "OK", "SHA-256 matches the download manifest"
                    verified += 1
                else:
                    status = "FAILED"
                    detail = "SHA-256 differs from the download manifest"
                    failed += 1
            else:
                unverified += 1

            print(f"   {status:<11} {path.name:<28} {size / 1e6:>9.2f} MB  {detail}")

        if not args.skip_structural:
            ok, message = structural_check(dataset, paths)
            print(f"   {'OK' if ok else 'FAILED':<11} {'structural check':<28} "
                  f"{'':>9}     {message}")
            if not ok:
                failed += 1

    print(f"\n{verified} file(s) checksum-verified, {unverified} unverified, "
          f"{failed} problem(s).")
    if failed:
        print("A FAILED entry means the bytes on disk are not the ones the publisher "
              "released. Re-download before using them.", file=sys.stderr)
        return 1
    if unverified and args.strict:
        print("Unverified files present and --strict was given.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
