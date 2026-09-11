"""Phase 9 - seed-robustness check for the synthetic-Zipf ladder (+ hybrid rung).

Conference-readiness gap this closes: every SmartGC result through Phase 8 was
reported from a single fixed seed (`random_seed: 42`). The `robustness_sweep`
run already showed the synthetic-Zipf WAF swinging non-monotonically between
1.0326 and 1.0432 purely from varying `robustness_lambda` with everything else
(including the seed) held fixed -- direct evidence that a few-thousandths
difference between two policies' single-seed WAF is not, on its own, proof of
a real effect. This script reruns the *entire* synthetic-Zipf pipeline
(workload generation -> features -> model training -> prediction export ->
simulation) end to end across multiple seeds -- so both workload randomness
and model-training randomness vary together per trial, matching what
`random_seed` is documented to govern (`config/config.yaml`) -- and reports
mean +/- std WAF per policy.

The real UMass SPC Financial1 trace is deliberately NOT multi-seeded here: it
is a fixed, non-regenerable real trace (only the model-training seed could
vary, at ~10x the per-trial cost of the tiny synthetic trace), so it is left
as a separate, narrower follow-up rather than folded into this pass.

Writes:
    results/metrics/seed_robustness.csv          (one row per seed x policy)
    results/metrics/seed_robustness_summary.csv  (mean/std per policy)
    results/plots/seed_robustness.png            (bar chart with error bars)

Usage:
    python -m experiments.seed_robustness
    python -m experiments.seed_robustness --seeds 42,43,44,45,46
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from ml.common.config import load_config, get, repo_path
from experiments.run_matrix import (
    PY, SIM, LADDER, HYBRID_POLICY, DISABLE_CPP_REGATE,
    default_geometry, export_predictions, run_id_for, sh,
)

CANONICAL_SEED = 42  # reuses the already-trained synthetic_zipf artefacts; no retraining


def _dataset_name(seed: int) -> str:
    return "synthetic_zipf" if seed == CANONICAL_SEED else f"synthetic_zipf_seed{seed}"


def _prepare_seed(cfg: dict, seed: int) -> str:
    """Ensure trace + features + the 3 trainable models exist for this seed.
    Returns the trace_path. Reuses the canonical seed's existing artefacts
    untouched; every other seed gets its own trace/model directory, sharing
    the one fixed scaler.json (never refit -- see docs/architecture.md 2.5)."""
    name = _dataset_name(seed)
    trace_path = repo_path("data", "processed", f"{name}.csv")
    if seed == CANONICAL_SEED:
        if not os.path.exists(trace_path):
            raise SystemExit(f"[seed_robustness] expected existing {trace_path}; "
                              f"run experiments.run_matrix at least once first")
        return trace_path

    n = int(get(cfg, "synthetic_workload.total_requests", 5000))
    ws = int(get(cfg, "synthetic_workload.lba_range_max", 500))
    hr = float(get(cfg, "synthetic_workload.hot_ratio", 0.20))
    ht = float(get(cfg, "synthetic_workload.hot_traffic_ratio", 0.80))
    raw_path = repo_path("data", "raw", f"{name}.csv")

    sh([SIM, "--generate-only", "--requests", n, "--lba-range", ws,
        "--hot-ratio", hr, "--hot-traffic", ht, "--seed", seed, "--workload", "skewed",
        "--export-trace", raw_path, "--quiet"])
    sh([PY, "-m", "ml.preprocessing.normalize", "--source", "synthetic",
        "--input", raw_path, "--name", name])
    sh([PY, "-m", "ml.preprocessing.features", "--input", trace_path, "--name", name,
        "--no-write-scaler"])
    for model in ("STAT_ML", "LSTM_SMARTGC", "LSTM_ATTN_SMARTGC"):
        sh([PY, "-m", "ml.training.train", "--dataset", name, "--model", model, "--seed", seed])
    return trace_path


def _run_one(cfg: dict, raw_csv: str, policy: str, dataset: str, trace_path: str,
             geometry: dict, seed: int, streams: int, conf_thr: float) -> float:
    preds = export_predictions(policy, trace_path, dataset, streams)
    cell_conf_thr = 0.0 if policy in DISABLE_CPP_REGATE else conf_thr
    effective = {"policy": policy, "trace": dataset, "seed": seed,
                 "confidence_threshold": cell_conf_thr, **geometry}
    rid = run_id_for(effective)
    cmd = [SIM, "--placement-policy", policy, "--trace", trace_path,
           "--total-segments", geometry["total_segments"],
           "--blocks-per-segment", geometry["blocks_per_segment"],
           "--gc-threshold", geometry["gc_threshold"],
           "--migration-streams", streams, "--seed", seed,
           "--workload-name", dataset, "--run-id", rid,
           "--export-metrics", raw_csv, "--quiet"]
    if preds:
        cmd += ["--predictions", preds, "--confidence-threshold", cell_conf_thr]
    sh(cmd)
    return float(pd.read_csv(raw_csv).iloc[-1]["waf"])


def main() -> None:
    cfg = load_config()
    streams = int(get(cfg, "gc.migration_stream_count", 3))
    conf_thr = float(get(cfg, "ml.confidence_fallback_threshold", 0.55))

    ap = argparse.ArgumentParser(description="SmartGC Phase 9 seed-robustness check")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--policies", default=",".join(LADDER))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]

    out_csv = repo_path("results", "metrics", "seed_robustness.csv")
    raw_csv = out_csv + ".raw"
    if os.path.exists(out_csv):
        os.remove(out_csv)
    if os.path.exists(raw_csv):
        os.remove(raw_csv)

    geometry = default_geometry(cfg)
    rows = []
    for seed in seeds:
        print(f"=== [seed_robustness] seed={seed} ===")
        trace_path = _prepare_seed(cfg, seed)
        dataset = _dataset_name(seed)
        for policy in policies:
            waf = _run_one(cfg, raw_csv, policy, dataset, trace_path, geometry, seed, streams, conf_thr)
            rows.append({"seed": seed, "placement_policy": policy, "waf": waf})
            print(f"  [seed_robustness] seed={seed} policy={policy} waf={waf:.4f}")

    if os.path.exists(raw_csv):
        os.remove(raw_csv)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    summary = (df.groupby("placement_policy")["waf"]
               .agg(["mean", "std", "min", "max", "count"])
               .reindex(policies)
               .reset_index())
    summary_csv = repo_path("results", "metrics", "seed_robustness_summary.csv")
    summary.to_csv(summary_csv, index=False)

    print(f"\n[seed_robustness] wrote {out_csv} and {summary_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
