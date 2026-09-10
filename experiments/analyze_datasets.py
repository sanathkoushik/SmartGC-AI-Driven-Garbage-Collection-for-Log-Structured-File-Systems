#!/usr/bin/env python3
"""Build the dataset inventory and apply the frozen workload-selection rules.

    python experiments/analyze_datasets.py            # inventory + role assignment
    python experiments/analyze_datasets.py --sources  # also regenerate the source comparison

Reads every ``results/dataset_statistics/*.json`` produced by
``ml.preprocessing.normalize`` and writes:

    results/dataset_inventory.csv          measured statistics, one row per workload
    results/dataset_roles.json             pretrain / target / external assignment
    results/dataset_source_comparison.csv  (with --sources) the source evaluation

The role assignment comes from ``ml.dataset_selection``, whose thresholds are
fixed in advance and depend on no measured model or WAF result. Running this
before training is what makes "we did not cherry-pick the workload" checkable
rather than merely asserted.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.dataset_selection import SELECTION_CRITERIA, assign_roles, check_eligibility  # noqa: E402

STATS_DIR = REPO_ROOT / "results" / "dataset_statistics"
INVENTORY_PATH = REPO_ROOT / "results" / "dataset_inventory.csv"
ROLES_PATH = REPO_ROOT / "results" / "dataset_roles.json"
SOURCE_COMPARISON_PATH = REPO_ROOT / "results" / "dataset_source_comparison.csv"

INVENTORY_COLUMNS = [
    "trace_id", "dataset", "device_selected",
    "raw_records_read", "records_accepted", "blocks_emitted",
    "block_requests", "write_blocks", "read_blocks", "write_percentage",
    "multi_block_requests", "unaligned_requests",
    "unique_lbas", "unique_written_lbas", "lba_min", "lba_max", "lba_span",
    "total_bytes", "mean_request_size_bytes",
    "duration_seconds",
    "rewrite_events", "rewrite_ratio",
    "lbas_written_more_than_once", "repeated_write_lba_fraction",
    "max_writes_to_single_lba", "top1pct_lba_write_share",
    "median_rewrite_interval_us", "mean_rewrite_interval_us",
    "p90_rewrite_interval_us", "p99_rewrite_interval_us",
    "dropped_total", "dropped_malformed", "dropped_bad_offset", "dropped_bad_size",
    "dropped_oversized_request", "dropped_unknown_operation", "dropped_other_device",
    "out_of_order_records", "duplicate_timestamps", "truncated_by_limit",
    "eligible", "ineligible_reasons", "normalized_path",
]


def load_statistics(stats_dir: Path = STATS_DIR) -> list[dict[str, Any]]:
    if not stats_dir.is_dir():
        return []
    records = []
    for path in sorted(stats_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            records.append(json.load(handle))
    return records


def write_inventory(statistics: list[dict[str, Any]], path: Path = INVENTORY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for record in sorted(statistics, key=lambda item: str(item.get("trace_id", ""))):
            eligibility = check_eligibility(record)
            row = dict(record)
            row["eligible"] = eligibility.eligible
            row["ineligible_reasons"] = "; ".join(eligibility.reasons)
            writer.writerow(row)
    print(f"Wrote {path.relative_to(REPO_ROOT)}  ({len(statistics)} workloads)")


def write_roles(statistics: list[dict[str, Any]], path: Path = ROLES_PATH) -> dict[str, Any]:
    roles = assign_roles(statistics)
    payload = roles.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {path.relative_to(REPO_ROOT)}")
    return payload


def write_source_comparison(path: Path = SOURCE_COMPARISON_PATH) -> None:
    from experiments.dataset_sources import CSV_COLUMNS, SOURCES

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in SOURCES:
            writer.writerow(record.to_dict())
    print(f"Wrote {path.relative_to(REPO_ROOT)}  ({len(SOURCES)} candidate sources)")


def print_summary(statistics: list[dict[str, Any]], roles: dict[str, Any]) -> None:
    print("\nMeasured workload inventory")
    print(f"  {'trace':<22} {'writes':>12} {'uniq LBA':>10} {'rewrite':>8} "
          f"{'dur (s)':>10} {'top1%':>7}  eligible")
    for record in sorted(statistics, key=lambda item: str(item.get("trace_id", ""))):
        eligibility = check_eligibility(record)
        print(f"  {str(record.get('trace_id', '?')):<22} "
              f"{int(record.get('write_blocks', 0) or 0):>12,} "
              f"{int(record.get('unique_written_lbas', 0) or 0):>10,} "
              f"{float(record.get('rewrite_ratio', 0.0) or 0.0):>8.3f} "
              f"{float(record.get('duration_seconds', 0.0) or 0.0):>10,.0f} "
              f"{float(record.get('top1pct_lba_write_share', 0.0) or 0.0):>7.3f}  "
              f"{'yes' if eligibility.eligible else 'NO: ' + '; '.join(eligibility.reasons)}")

    print("\nFrozen selection criteria (docs/methodology.md)")
    for key, value in SELECTION_CRITERIA.items():
        print(f"  {key}: {value}")

    print("\nRole assignment")
    print(f"  pretrain : {', '.join(roles['pretrain']) or '(none)'}")
    print(f"  targets  : {', '.join(roles['targets']) or '(none)'}")
    print(f"  external : {', '.join(roles['external']) or '(none)'}")
    if roles["excluded"]:
        print(f"  excluded : {len(roles['excluded'])} workload(s) failed eligibility")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", action="store_true",
                        help="also regenerate results/dataset_source_comparison.csv")
    parser.add_argument("--stats-dir", type=Path, default=STATS_DIR)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.sources:
        write_source_comparison()

    statistics = load_statistics(args.stats_dir)
    if not statistics:
        print(f"No dataset statistics found under {args.stats_dir}.", file=sys.stderr)
        print("Run: python -m ml.preprocessing.normalize --discover", file=sys.stderr)
        return 2

    write_inventory(statistics)
    roles = write_roles(statistics)
    print_summary(statistics, roles)

    if not roles["targets"]:
        print("\nWARNING: no eligible MSR target workload was found; "
              "the fine-tuning experiments cannot run.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
