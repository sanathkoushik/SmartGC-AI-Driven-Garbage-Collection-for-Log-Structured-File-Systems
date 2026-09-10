"""Convert raw MSR / SYSTOR / FIU traces into the normalized SmartGC contract.

    python -m ml.preprocessing.normalize --discover
    python -m ml.preprocessing.normalize --dataset msr --input data/raw/msr/hm_0.csv.gz

Writes one CSV per workload to ``data/processed/normalized/`` and one statistics
JSON per workload to ``results/dataset_statistics/``.

A normalized trace always describes exactly **one** block device. See
``ml/preprocessing/base_parser.py`` for why: LBAs are only meaningful within a
volume, so pooling devices would fabricate rewrites.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ml.config import REPO_ROOT, load_config
from ml.preprocessing.parsers import DATASETS, ParseStats, get_parser

NORMALIZED_DIR = REPO_ROOT / "data" / "processed" / "normalized"
STATS_DIR = REPO_ROOT / "results" / "dataset_statistics"
RAW_DIR = REPO_ROOT / "data" / "raw"

_NON_TRACE_NAMES = {"readme.txt", "readme.md", "md5.txt", "disclaimer.txt",
                    ".gitkeep", "download_manifest.json"}
_ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tar.xz", ".tgz", ".zip")
# ".part" is an in-progress download; treating one as a trace would silently
# feed a truncated file into the pipeline.
_INCOMPLETE_SUFFIXES = (".part", ".tmp", ".crdownload")
# SYSTOR ships one file per hour per LUN: 2016022207-LUN0.csv.gz
_SYSTOR_NAME = re.compile(r"^(?P<stamp>\d{10})-(?P<lun>LUN\d+)\.csv(\.gz)?$", re.IGNORECASE)


def default_trace_id(dataset: str, path: Path) -> str:
    """Derive a stable workload identifier from a raw trace filename."""
    stem = path.name
    for suffix in (".gz", ".bz2", ".xz", ".lzma"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    stem = Path(stem).stem
    return f"{dataset.lower()}_{stem.replace(' ', '_')}"


def discover_raw_traces(raw_dir: Path = RAW_DIR) -> list[tuple[str, str, list[Path]]]:
    """Group raw files under data/raw/<dataset>/ into workloads.

    Returns (dataset, trace_id, files). SYSTOR hourly shards belonging to the
    same LUN are grouped into one chronological workload; every other dataset
    ships one file per volume.
    """
    workloads: list[tuple[str, str, list[Path]]] = []

    for dataset in DATASETS:
        dataset_dir = raw_dir / dataset
        if not dataset_dir.is_dir():
            continue
        candidates = [p for p in sorted(dataset_dir.rglob("*"))
                      if p.is_file()
                      and p.name.lower() not in _NON_TRACE_NAMES
                      and not p.name.lower().endswith(_ARCHIVE_SUFFIXES)
                      and not p.name.lower().endswith(_INCOMPLETE_SUFFIXES)]

        if dataset == "systor":
            by_lun: dict[str, list[Path]] = {}
            for path in candidates:
                match = _SYSTOR_NAME.match(path.name)
                if not match:
                    continue
                by_lun.setdefault(match.group("lun").upper(), []).append(path)
            for lun, paths in sorted(by_lun.items()):
                # Filenames embed YYYYMMDDHH, so lexical order is chronological.
                workloads.append((dataset, f"systor_{lun.lower()}", sorted(paths)))
            continue

        for path in candidates:
            workloads.append((dataset, default_trace_id(dataset, path), [path]))

    return workloads


def compute_trace_statistics(table: pd.DataFrame, stats: ParseStats) -> dict[str, Any]:
    """Measure the workload properties used for subset selection.

    Counts are over normalized per-block records, so "block requests" means
    block-level accesses after multi-block expansion. Raw request counts and
    sizes are reported separately from the parser's audit trail.
    """
    result: dict[str, Any] = stats.to_dict()

    empty_defaults = {
        "block_requests": 0, "write_blocks": 0, "read_blocks": 0,
        "write_percentage": 0.0, "unique_lbas": 0, "unique_written_lbas": 0,
        "lba_min": 0, "lba_max": 0, "lba_span": 0, "duration_seconds": 0.0,
        "rewrite_events": 0, "rewrite_ratio": 0.0, "lbas_written_more_than_once": 0,
        "repeated_write_lba_fraction": 0.0, "max_writes_to_single_lba": 0,
        "top1pct_lba_write_share": 0.0, "mean_request_size_bytes": 0.0,
        "median_rewrite_interval_us": 0.0, "mean_rewrite_interval_us": 0.0,
    }
    if table.empty:
        result.update(empty_defaults)
        return result

    operations = table["operation"].to_numpy()
    writes = table[operations == "W"]
    n = len(table)

    result["block_requests"] = n
    result["write_percentage"] = 100.0 * len(writes) / n if n else 0.0
    result["unique_lbas"] = int(table["lba"].nunique())
    result["lba_min"] = int(table["lba"].min())
    result["lba_max"] = int(table["lba"].max())
    result["lba_span"] = result["lba_max"] - result["lba_min"] + 1
    result["duration_seconds"] = stats.duration_microseconds / 1e6

    accepted = max(stats.records_accepted, 1)
    result["mean_request_size_bytes"] = stats.total_bytes / accepted

    if len(writes):
        written = writes["lba"].to_numpy()
        codes, _ = pd.factorize(written)
        counts = np.bincount(codes)
        unique_written = int(counts.size)

        result["unique_written_lbas"] = unique_written
        # A rewrite is any write to an LBA that has been written before, so the
        # count is total writes minus the first write of each distinct LBA.
        result["rewrite_events"] = int(len(writes) - unique_written)
        result["rewrite_ratio"] = result["rewrite_events"] / len(writes)
        result["lbas_written_more_than_once"] = int((counts > 1).sum())
        result["repeated_write_lba_fraction"] = (
            result["lbas_written_more_than_once"] / unique_written if unique_written else 0.0
        )
        result["max_writes_to_single_lba"] = int(counts.max())

        # Concentration of write traffic: the share taken by the busiest 1% of
        # written LBAs. Skew is what hot/cold separation can exploit.
        order = np.sort(counts)[::-1]
        top = max(1, int(round(0.01 * order.size)))
        result["top1pct_lba_write_share"] = float(order[:top].sum() / counts.sum())

        # Rewrite-interval summary, computed the same way as the model targets.
        from ml.preprocessing.sequences import rewrite_intervals

        intervals = rewrite_intervals(writes["lba"].to_numpy(),
                                      writes["timestamp"].to_numpy())["interval"]
        if intervals.size:
            result["median_rewrite_interval_us"] = float(np.median(intervals))
            result["mean_rewrite_interval_us"] = float(np.mean(intervals))
            result["p90_rewrite_interval_us"] = float(np.percentile(intervals, 90))
            result["p99_rewrite_interval_us"] = float(np.percentile(intervals, 99))
        else:
            result["median_rewrite_interval_us"] = 0.0
            result["mean_rewrite_interval_us"] = 0.0
    else:
        result.update({k: v for k, v in empty_defaults.items()
                       if k not in ("block_requests", "write_percentage", "unique_lbas",
                                    "lba_min", "lba_max", "lba_span", "duration_seconds",
                                    "mean_request_size_bytes")})

    return result


def normalize_trace(dataset: str,
                    inputs: Path | list[Path],
                    trace_id: str | None = None,
                    max_records: int | None = None,
                    output_dir: Path = NORMALIZED_DIR,
                    stats_dir: Path = STATS_DIR,
                    block_size_bytes: int = 4096,
                    max_blocks_per_request: int = 2048,
                    max_offset_bytes: int = 1 << 50,
                    device: str = "auto",
                    write_output: bool = True) -> dict[str, Any]:
    """Normalize one workload (one or more raw files) and write its outputs."""
    paths = [inputs] if isinstance(inputs, Path) else list(inputs)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Raw trace not found: {path}")

    trace_id = trace_id or default_trace_id(dataset, paths[0])
    parser = get_parser(
        dataset,
        block_size_bytes=block_size_bytes,
        max_blocks_per_request=max_blocks_per_request,
        max_offset_bytes=max_offset_bytes,
        device=device,
    )

    table, stats = parser.parse(paths, trace_id=trace_id, max_records=max_records)
    statistics = compute_trace_statistics(table, stats)
    statistics["block_size_bytes"] = block_size_bytes
    statistics["timestamp_unit"] = "microseconds_since_trace_start"

    if write_output:
        output_dir.mkdir(parents=True, exist_ok=True)
        stats_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{trace_id}.csv"
        table.to_csv(out_path, index=False)
        statistics["normalized_path"] = str(out_path.relative_to(REPO_ROOT)).replace("\\", "/")
        with (stats_dir / f"{trace_id}.json").open("w", encoding="utf-8") as handle:
            json.dump(statistics, handle, indent=2, sort_keys=True, default=str)

    return statistics


def _print_summary(statistics: dict[str, Any]) -> None:
    print(f"  trace_id            : {statistics['trace_id']}")
    print(f"  device              : {statistics.get('device_selected') or '(single)'}")
    print(f"  raw records read    : {statistics['raw_records_read']:,}")
    print(f"  requests accepted   : {statistics['records_accepted']:,}"
          f"  (multi-block {statistics['multi_block_requests']:,},"
          f" unaligned {statistics['unaligned_requests']:,})")
    print(f"  blocks emitted      : {statistics['blocks_emitted']:,}")
    print(f"  dropped (total)     : {statistics['dropped_total']:,}"
          f"  [malformed {statistics['dropped_malformed']:,},"
          f" offset {statistics['dropped_bad_offset']:,},"
          f" size {statistics['dropped_bad_size']:,},"
          f" oversized {statistics['dropped_oversized_request']:,},"
          f" unknown-op {statistics['dropped_unknown_operation']:,},"
          f" other-device {statistics['dropped_other_device']:,}]")
    print(f"  write / read blocks : {statistics['write_blocks']:,} / {statistics['read_blocks']:,}"
          f"  ({statistics['write_percentage']:.1f}% writes)")
    print(f"  unique LBAs         : {statistics['unique_lbas']:,}"
          f"  (written {statistics['unique_written_lbas']:,})")
    print(f"  rewrite events      : {statistics['rewrite_events']:,}"
          f"  (ratio {statistics['rewrite_ratio']:.3f})")
    print(f"  duration            : {statistics['duration_seconds']:,.1f} s")
    if statistics.get("truncated_by_limit"):
        print("  NOTE: truncated by --max-records (chronological prefix)")
    if statistics.get("unknown_operation_examples"):
        print(f"  unknown op values   : {statistics['unknown_operation_examples']}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=list(DATASETS), help="raw trace format")
    parser.add_argument("--input", type=Path, nargs="*", help="raw trace file(s)")
    parser.add_argument("--trace-id", help="override the derived workload identifier")
    parser.add_argument("--discover", action="store_true",
                        help="normalize every workload found under data/raw/<dataset>/")
    parser.add_argument("--max-records", type=int, default=None,
                        help="cap on raw records consumed (chronological prefix)")
    parser.add_argument("--device", default="auto",
                        help="'auto' (require one device), 'dominant', or an explicit device key")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config(args.config)
    common = {
        "max_records": args.max_records,
        "block_size_bytes": config.simulator.block_size_bytes,
        "max_blocks_per_request": config.preprocessing.max_blocks_per_request,
        "max_offset_bytes": config.preprocessing.max_offset_bytes,
        "device": args.device,
    }

    if args.discover:
        targets = discover_raw_traces()
        if not targets:
            print(f"No raw traces found under {RAW_DIR}.", file=sys.stderr)
            print("Run: python scripts/download_datasets.py --dataset all", file=sys.stderr)
            return 2
    else:
        if not args.dataset or not args.input:
            parser.error("--dataset and --input are required unless --discover is given")
        targets = [(args.dataset, args.trace_id or default_trace_id(args.dataset, args.input[0]),
                    list(args.input))]

    failures = 0
    for dataset, trace_id, paths in targets:
        label = paths[0].name if len(paths) == 1 else f"{len(paths)} files"
        print(f"\nNormalizing [{dataset}] {trace_id}  ({label})")
        try:
            statistics = normalize_trace(dataset, paths, trace_id=trace_id, **common)
        except Exception as error:                  # noqa: BLE001 - reported, not swallowed
            print(f"  FAILED: {type(error).__name__}: {error}", file=sys.stderr)
            failures += 1
            continue
        _print_summary(statistics)

    if failures:
        print(f"\n{failures} workload(s) failed to normalize.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
