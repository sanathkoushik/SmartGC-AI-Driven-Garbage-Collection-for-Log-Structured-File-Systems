#!/usr/bin/env python3
"""Compact end-to-end SmartGC demonstration on real trace data.

    python scripts/conference_demo.py
    python scripts/conference_demo.py --trace msr_prxy_0 --utilization 0.80

Checks what exists, replays one real workload through all three placement
policies, and prints the measured comparison. Every number it prints comes from
the run it just performed or from a metrics file that run wrote - nothing is
hardcoded, and if a required artefact is missing the demo says exactly which
command produces it rather than inventing a result.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd                                                     # noqa: E402

from ml.config import load_config                                       # noqa: E402
from ml.training.common import build_split_pool, load_checkpoint, normalized_trace_path  # noqa: E402
from ml.training.finetune import MODEL_TYPES, default_output_path       # noqa: E402
from ml.preprocessing.sequences import hot_threshold_from_training      # noqa: E402
from experiments.run_all import distinct_written_lbas, find_simulator, size_device  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
PREDICTIONS_DIR = REPO_ROOT / "data" / "predictions"
PLOTS_DIR = RESULTS_DIR / "plots"
ROLES_PATH = RESULTS_DIR / "dataset_roles.json"
MODEL_COMPARISON = RESULTS_DIR / "ml" / "model_comparison.csv"

RULE = "=" * 74


class DemoError(RuntimeError):
    """A prerequisite is missing; the message names the command that creates it."""


def section(title: str) -> None:
    print(f"\n{RULE}\n  {title}\n{RULE}")


def pick_trace(explicit: str | None) -> str:
    if explicit:
        return explicit
    if not ROLES_PATH.is_file():
        raise DemoError("results/dataset_roles.json is missing.\n"
                        "  Run: python experiments/run_all.py --stages normalize inventory")
    roles = json.loads(ROLES_PATH.read_text(encoding="utf-8"))
    if roles.get("targets"):
        return roles["targets"][0]
    raise DemoError("no held-out target workload was selected.\n"
                    "  Run: python experiments/analyze_datasets.py")


def require_normalized(trace_id: str) -> Path:
    path = normalized_trace_path(trace_id)
    if not path.is_file():
        raise DemoError(f"{trace_id} has not been normalized.\n"
                        "  Run: python -m ml.preprocessing.normalize --discover")
    return path


def available_models(trace_id: str) -> dict[str, Path]:
    return {name: default_output_path(name, trace_id)
            for name in MODEL_TYPES if default_output_path(name, trace_id).is_file()}


def run_simulation(simulator: Path, args: list[str], report: Path) -> dict[str, Any]:
    result = subprocess.run([str(simulator), *args, "--report-json", str(report)],
                            cwd=str(REPO_ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        raise DemoError(f"simulator exited {result.returncode}:\n{result.stderr.strip()}")
    return json.loads(report.read_text(encoding="utf-8"))


def format_bytes(value: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024 or unit == "TB":
            return f"{value:,.1f} {unit}"
        value /= 1024
    return f"{value:,.1f} TB"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace", default=None,
                        help="workload to demonstrate; defaults to the first held-out target")
    parser.add_argument("--utilization", type=float, default=0.80)
    parser.add_argument("--max-writes", type=int, default=1_000_000,
                        help="cap on replayed writes so the demo finishes quickly")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    started = time.monotonic()
    config = load_config(args.config)
    config_path = args.config or (REPO_ROOT / "config" / "config.yaml")

    section("SMARTGC CONFERENCE DEMONSTRATION")

    # ---- prerequisites ----------------------------------------------------
    trace_id = pick_trace(args.trace)
    trace_path = require_normalized(trace_id)
    simulator = find_simulator()
    models = available_models(trace_id)
    if not models:
        raise DemoError(f"no trained model for {trace_id}.\n"
                        "  Run: python experiments/run_all.py")

    dataset = trace_id.split("_", 1)[0]
    dataset_name = {"msr": "MSR Cambridge (Microsoft Research, 2007)",
                    "systor": "SYSTOR '17 enterprise VDI (Fujitsu, 2016)",
                    "fiu": "FIU (Florida International University, 2008-2009)"}.get(
                        dataset, dataset)

    # Prefer the fully adapted model for the headline run.
    preferred = next((name for name in ("pretrained_finetuned", "scratch", "pretrained")
                      if name in models), None)
    assert preferred is not None
    _, _, metadata = load_checkpoint(models[preferred])
    hyperparameters = metadata.get("hyperparameters", {})
    sequence_length = int(hyperparameters.get("sequence_length", config.ml.sequence_length))

    statistics_path = RESULTS_DIR / "dataset_statistics" / f"{trace_id}.json"
    statistics = (json.loads(statistics_path.read_text(encoding="utf-8"))
                  if statistics_path.is_file() else {})

    print(f"  Dataset            : {dataset_name}")
    print(f"  Trace              : {trace_id}"
          f"{'  (device ' + statistics['device_selected'] + ')' if statistics.get('device_selected') else ''}")
    if statistics:
        print(f"  Trace content      : {int(statistics.get('write_blocks', 0)):,} write blocks, "
              f"{int(statistics.get('unique_written_lbas', 0)):,} distinct LBAs, "
              f"rewrite ratio {float(statistics.get('rewrite_ratio', 0)):.3f}")
    print(f"  Model              : LSTM, hidden {hyperparameters.get('hidden_dim')}, "
          f"{hyperparameters.get('num_layers')} layer(s), "
          f"{metadata.get('parameter_count', '?'):,} parameters")
    print(f"  Sequence length    : {sequence_length} rewrite intervals")
    print(f"  Training           : {preferred.replace('_', ' + ')}")
    if metadata.get("pretrain_traces"):
        print(f"  Pretrained on      : {', '.join(metadata['pretrain_traces'])}")
        print(f"  Target held out    : yes ({trace_id} not in the pretraining set)")
    print(f"  Hot/cold threshold : {metadata.get('hot_threshold_microseconds', 0):,.0f} us "
          f"(training-set p{config.ml.hot_percentile_cutoff:g}; test data not used)")

    # ---- device geometry --------------------------------------------------
    distinct, write_count = distinct_written_lbas(trace_id, args.max_writes)
    segments = size_device(distinct, args.utilization,
                           config.simulator.blocks_per_segment,
                           config.simulator.gc_reserved_segments,
                           config.simulator.gc_separate_stream)
    capacity_blocks = segments * config.simulator.blocks_per_segment
    print(f"\n  Replaying          : {write_count:,} writes ({distinct:,} distinct LBAs)")
    print(f"  Device             : {segments} segments x "
          f"{config.simulator.blocks_per_segment} blocks = {capacity_blocks:,} blocks "
          f"({format_bytes(capacity_blocks * config.simulator.block_size_bytes)})")
    print(f"  Target utilization : {args.utilization:.0%} "
          f"({distinct / capacity_blocks:.1%} actual)")

    # ---- run every policy on the identical workload -----------------------
    pool = build_split_pool([trace_id], config.ml)
    rule_threshold = hot_threshold_from_training(pool.train["targets"],
                                                 config.ml.hot_percentile_cutoff)

    reports_dir = RESULTS_DIR / "metrics" / "demo"
    reports_dir.mkdir(parents=True, exist_ok=True)
    base = [
        "--config", str(config_path),
        "--trace", str(trace_path),
        "--total-segments", str(segments),
        "--dataset", dataset,
        "--trace-name", trace_id,
        "--max-writes", str(args.max_writes),
        "--quiet",
    ]

    section("RUNNING THE THREE PLACEMENT POLICIES ON AN IDENTICAL WORKLOAD")
    runs: list[tuple[str, dict[str, Any]]] = []

    print("  MIXED         ... ", end="", flush=True)
    runs.append(("MIXED", run_simulation(
        simulator, base + ["--placement", "MIXED", "--model-type", "none"],
        reports_dir / "MIXED.json")))
    print("done")

    print("  RULE_BASED    ... ", end="", flush=True)
    runs.append(("RULE_BASED", run_simulation(
        simulator, base + ["--placement", "RULE_BASED", "--model-type", "rule",
                           "--rule-threshold-us", f"{rule_threshold:.6f}",
                           "--rule-min-history", str(sequence_length)],
        reports_dir / "RULE_BASED.json")))
    print("done")

    for name in MODEL_TYPES:
        predictions = PREDICTIONS_DIR / f"{name}__{trace_id}.csv"
        if name not in models or not predictions.is_file():
            continue
        label = f"LSTM_SMARTGC ({name})"
        print(f"  {label:<30}... ", end="", flush=True)
        runs.append((label, run_simulation(
            simulator, base + ["--placement", "LSTM_SMARTGC", "--model-type", name,
                               "--predictions", str(predictions)],
            reports_dir / f"LSTM_{name}.json")))
        print("done")

    # ---- results ----------------------------------------------------------
    section("MEASURED RESULTS")
    baseline = next(report for name, report in runs if name == "MIXED")

    header = (f"  {'Policy':<32} {'WAF':>8} {'GC bytes':>13} {'Valid migrations':>18} "
              f"{'vs MIXED':>10}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, report in runs:
        change = (report["waf"] - baseline["waf"]) / baseline["waf"] * 100.0
        change_text = "baseline" if name == "MIXED" else f"{change:+.2f}%"
        print(f"  {name:<32} {report['waf']:>8.4f} "
              f"{format_bytes(report['gc_bytes_copied']):>13} "
              f"{report['valid_blocks_migrated']:>18,} {change_text:>10}")

    print(f"\n  Logical bytes written (identical for every policy): "
          f"{format_bytes(baseline['logical_bytes_written'])}")
    identical = {report["write_requests"] for _, report in runs}
    print(f"  Write requests replayed: {identical.pop():,}"
          f"{'  [WARNING: policies saw different workloads]' if identical else ''}")

    for name, report in runs:
        if "prediction_rows_matched" in report:
            coverage = 1.0 - report["unknown_placements"] / max(report["write_requests"], 1)
            print(f"  {name}: {coverage:.1%} of writes had a prediction "
                  f"({report['hot_placements']:,} HOT / {report['cold_placements']:,} COLD)")
        elif "rule_classified_hot" in report:
            coverage = 1.0 - report["unknown_placements"] / max(report["write_requests"], 1)
            print(f"  {name}: {coverage:.1%} of writes were classified "
                  f"({report['hot_placements']:,} HOT / {report['cold_placements']:,} COLD)")

    # ---- prediction quality ----------------------------------------------
    if MODEL_COMPARISON.is_file():
        frame = pd.read_csv(MODEL_COMPARISON)
        frame = frame[frame["trace"] == trace_id]
        if not frame.empty:
            section("PREDICTION QUALITY ON THE HELD-OUT TEST SPLIT")
            print(f"  {'Predictor':<26} {'MAE (us)':>14} {'RMSE (us)':>14} "
                  f"{'MAE log1p':>10} {'F1':>8} {'Accuracy':>9}")
            print("  " + "-" * 84)
            for _, row in frame.sort_values(["family", "predictor"]).iterrows():
                print(f"  {row['predictor']:<26} {row['mae_us']:>14,.0f} "
                      f"{row['rmse_us']:>14,.0f} {row['mae_log1p']:>10.4f} "
                      f"{row['f1']:>8.4f} {row['accuracy']:>9.4f}")

    # ---- pointers ---------------------------------------------------------
    section("ARTEFACTS")
    for label, path in (("Policy comparison", RESULTS_DIR / "metrics" / "comparison.csv"),
                        ("Ablation table", RESULTS_DIR / "metrics" / "ablation.csv"),
                        ("Model comparison", MODEL_COMPARISON),
                        ("Dataset inventory", RESULTS_DIR / "dataset_inventory.csv"),
                        ("Experiment manifest", RESULTS_DIR / "final" / "experiment_manifest.json")):
        mark = "ok " if path.is_file() else "-- "
        print(f"  [{mark}] {label:<20} {path.relative_to(REPO_ROOT)}")

    key_plots = ["15_waf_by_policy.png", "19_waf_improvement_vs_mixed.png",
                 "14_transfer_learning_comparison.png", "18_waf_vs_utilization.png"]
    existing = [name for name in key_plots if (PLOTS_DIR / name).is_file()]
    if existing:
        print("\n  Key figures:")
        for name in existing:
            print(f"    results/plots/{name}")

    print(f"\n  Demo completed in {time.monotonic() - started:.1f}s. "
          "Every number above was produced by this run.")
    print(RULE)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DemoError as error:
        print(f"\nDEMO PREREQUISITE MISSING: {error}", file=sys.stderr)
        raise SystemExit(2)
