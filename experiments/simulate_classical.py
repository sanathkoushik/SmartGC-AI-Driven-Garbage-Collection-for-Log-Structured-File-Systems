#!/usr/bin/env python3
"""Replay the classical-ML predictions through the LFS simulator.

    python experiments/simulate_classical.py --targets msr_rsrch_0 msr_src2_0 systor_lun0

For each target and utilization this uses the geometry, seed, write cap and
trace that ``experiments/run_all.py`` uses for the existing policies, so the new
rows are directly comparable with the MIXED / RULE_BASED / LSTM rows in
``results/metrics/comparison.csv``. A MIXED run is repeated as a control and its
WAF must equal the existing row, otherwise the geometry has drifted and the
comparison is not valid.

Writes ``results/metrics/classical_comparison.csv`` and
``results/final/waf_full_comparison.csv`` (every policy side by side).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_all import (                                    # noqa: E402
    COMPARISON_CSV,
    DEFAULT_UTILIZATIONS,
    PREDICTIONS_DIR,
    distinct_written_lbas,
    find_simulator,
    size_device,
)
from ml.config import load_config                                     # noqa: E402
from ml.inference.predict_classical import generate_classical_predictions  # noqa: E402
from ml.models.classical import CLASSICAL_MODEL_TYPES                 # noqa: E402
from ml.training.common import normalized_trace_path, repo_relative   # noqa: E402

CLASSICAL_SIM_CSV = REPO_ROOT / "results" / "metrics" / "classical_comparison.csv"
WAF_FULL_CSV = REPO_ROOT / "results" / "final" / "waf_full_comparison.csv"
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def run_simulator(command: list[str]) -> None:
    result = subprocess.run([str(part) for part in command], cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"simulator failed (exit {result.returncode}): {' '.join(map(str, command))}")


def simulate_target(trace_id: str, models: Iterable[str], utilizations: Iterable[float],
                    max_writes: int, seed: int) -> None:
    config = load_config(None)
    sim = config.simulator
    simulator = find_simulator()
    trace_path = normalized_trace_path(trace_id)
    distinct, write_count = distinct_written_lbas(trace_id, max_writes)
    dataset = trace_id.split("_", 1)[0]
    print(f"\n=== {trace_id}: {write_count:,} writes, {distinct:,} distinct LBAs ===")

    for model_type in models:
        generate_classical_predictions(trace_id, model_type)

    for utilization in utilizations:
        segments = size_device(distinct, utilization, sim.blocks_per_segment,
                               sim.gc_reserved_segments, sim.gc_separate_stream)
        base = [simulator, "--config", CONFIG_PATH, "--trace", trace_path,
                "--total-segments", segments, "--dataset", dataset, "--trace-name", trace_id,
                "--export-metrics", CLASSICAL_SIM_CSV, "--seed", seed, "--quiet"]
        if max_writes:
            base += ["--max-writes", max_writes]
        print(f"-- utilization {utilization:.0%}: {segments} segments")
        # Control: must reproduce the committed MIXED row exactly.
        run_simulator(base + ["--placement", "MIXED", "--model-type", "none"])
        for model_type in models:
            predictions = PREDICTIONS_DIR / f"{model_type}__{trace_id}.csv"
            run_simulator(base + ["--placement", "LSTM_SMARTGC", "--model-type", model_type,
                                  "--predictions", predictions])


def check_control_and_merge() -> pd.DataFrame:
    new = pd.read_csv(CLASSICAL_SIM_CSV)
    old = pd.read_csv(COMPARISON_CSV)
    key = ["trace", "total_segments"]

    control = new[new["policy"] == "MIXED"].merge(
        old[old["policy"] == "MIXED"][key + ["waf"]], on=key, suffixes=("", "_existing"))
    mismatched = control[(control["waf"] - control["waf_existing"]).abs() > 1e-9]
    if control.empty:
        print("WARNING: no MIXED control rows matched the existing comparison; "
              "geometry could not be cross-checked")
    elif not mismatched.empty:
        raise RuntimeError("MIXED control differs from results/metrics/comparison.csv:\n"
                           + mismatched[key + ["waf", "waf_existing"]].to_string())
    else:
        print(f"MIXED control reproduces the existing WAF in all {len(control)} cases")

    classical = new[new["policy"] != "MIXED"].copy()
    classical["configuration"] = classical["model_type"]
    existing = old.copy()

    def label(row: pd.Series) -> str:
        return (f"LSTM {row['model_type']}" if row["policy"] == "LSTM_SMARTGC" else row["policy"])
    existing["configuration"] = existing.apply(label, axis=1)

    combined = pd.concat([existing, classical], ignore_index=True)
    base = combined[combined["policy"] == "MIXED"].set_index(["trace", "total_segments"])["waf"]
    combined["mixed_waf"] = [base.get((t, s)) for t, s in zip(combined["trace"],
                                                              combined["total_segments"])]
    combined["waf_change_vs_mixed_pct"] = (combined["waf"] / combined["mixed_waf"] - 1.0) * 100
    return combined


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", default=list(CLASSICAL_MODEL_TYPES),
                        choices=list(CLASSICAL_MODEL_TYPES))
    parser.add_argument("--utilizations", type=float, nargs="*", default=list(DEFAULT_UTILIZATIONS))
    parser.add_argument("--max-writes", type=int, default=1_000_000,
                        help="must equal max_writes_per_run in results/final/experiment_manifest.json")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    seed = args.seed if args.seed is not None else load_config(None).random_seed
    if CLASSICAL_SIM_CSV.exists():
        CLASSICAL_SIM_CSV.unlink()
    for trace_id in args.targets:
        simulate_target(trace_id, args.models, args.utilizations, args.max_writes, seed)

    combined = check_control_and_merge()
    WAF_FULL_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(WAF_FULL_CSV, index=False)
    print(f"Wrote {repo_relative(CLASSICAL_SIM_CSV)} and {repo_relative(WAF_FULL_CSV)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        print(f"FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
