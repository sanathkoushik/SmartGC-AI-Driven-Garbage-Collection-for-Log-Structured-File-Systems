"""Phase 8 - robustness-lambda sweep for HYBRID_ROBUST_SMARTGC.

Sweeps ``ml.robustness_lambda`` (see ml/inference/robust_blend.py) across
[0, 1] on the synthetic Zipf and real (UMass SPC Financial1) traces, exporting
predictions and running the simulator at each point, to produce the empirical
consistency-robustness tradeoff curve that motivates a default value for
``config/config.yaml``. Requires an already-trained LSTM_ATTN_SMARTGC
checkpoint per trace (run ``experiments.run_matrix`` first, or
``ml.training.train`` directly) -- this script never trains.

Writes ``results/metrics/robustness_sweep.csv``:
    workload_name, robustness_lambda, waf, mean_trust_weight

Usage:
    python -m experiments.robustness_sweep
    python -m experiments.robustness_sweep --lambdas 0,0.25,0.5,0.75,1.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import pandas as pd

from ml.common.config import load_config, get, repo_path
from experiments.run_matrix import (
    PY, SIM, HYBRID_POLICY, default_geometry, geometry_for_trace, run_id_for,
)


def sh(cmd: list) -> str:
    import subprocess
    print("  $ " + " ".join(str(c) for c in cmd))
    return subprocess.check_output([str(c) for c in cmd], text=True)


def run_point(cfg: dict, out_csv: str, trace_name: str, trace_path: str, lam: float,
              geometry: dict, seed: int, streams: int) -> None:
    stem = os.path.splitext(os.path.basename(trace_path))[0]
    name = f"{HYBRID_POLICY}_lam{lam}_{stem}"
    model_dir = repo_path("ml", "models", f"LSTM_ATTN_SMARTGC_{stem}")
    export_cmd = [PY, "-m", "ml.inference.export", "--trace", trace_path,
                  "--model", HYBRID_POLICY, "--model-dir", model_dir, "--name", name,
                  "--streams", streams, "--robust-lambda", lam]
    summary = json.loads(sh(export_cmd))

    effective = {"policy": HYBRID_POLICY, "trace": trace_name, "lambda": lam,
                 "streams": streams, "seed": seed, **geometry}
    rid = run_id_for(effective)
    preds = repo_path("data", "predictions", f"{name}.csv")
    sim_cmd = [SIM, "--placement-policy", HYBRID_POLICY, "--trace", trace_path,
               "--total-segments", geometry["total_segments"],
               "--blocks-per-segment", geometry["blocks_per_segment"],
               "--gc-threshold", geometry["gc_threshold"],
               "--migration-streams", streams, "--seed", seed,
               "--workload-name", f"{trace_name}_lam{lam}", "--run-id", rid,
               "--predictions", preds, "--confidence-threshold", 0.0,
               "--export-metrics", out_csv + ".raw", "--quiet"]
    sh(sim_cmd)

    raw = pd.read_csv(out_csv + ".raw")
    waf = float(raw.iloc[-1]["waf"])
    row = {"workload_name": trace_name, "robustness_lambda": lam, "waf": waf,
           "mean_trust_weight": summary.get("mean_trust_weight")}
    header = not os.path.exists(out_csv)
    pd.DataFrame([row]).to_csv(out_csv, mode="a", header=header, index=False)
    print(f"  [robustness_sweep] {trace_name} lambda={lam} -> waf={waf:.4f} "
          f"mean_trust_weight={row['mean_trust_weight']}")


def main() -> None:
    cfg = load_config()
    seed = int(get(cfg, "random_seed", 42))
    ap = argparse.ArgumentParser(description="SmartGC Phase 8 robustness-lambda sweep")
    ap.add_argument("--lambdas", default="0.0,0.2,0.35,0.5,0.65,0.8,1.0")
    ap.add_argument("--traces", default=None, help="comma-separated dataset names "
                     "(default: synthetic_zipf + evaluation.real_trace_name if trained)")
    args = ap.parse_args()
    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip() != ""]

    real = str(get(cfg, "evaluation.real_trace_name", "financial1"))
    real_op = float(get(cfg, "evaluation.real_op_ratio", 0.66))
    streams = int(get(cfg, "gc.migration_stream_count", 3))

    if args.traces:
        names = [t.strip() for t in args.traces.split(",") if t.strip()]
    else:
        names = ["synthetic_zipf"]
        if os.path.isdir(repo_path("ml", "models", f"LSTM_ATTN_SMARTGC_{real}")):
            names.append(real)

    out_csv = repo_path("results", "metrics", "robustness_sweep.csv")
    if os.path.exists(out_csv):
        os.remove(out_csv)
    if os.path.exists(out_csv + ".raw"):
        os.remove(out_csv + ".raw")

    for trace_name in names:
        trace_path = repo_path("data", "processed", f"{trace_name}.csv")
        if not os.path.exists(trace_path):
            print(f"[robustness_sweep] skip {trace_name}: {trace_path} missing")
            continue
        geom = default_geometry(cfg) if trace_name == "synthetic_zipf" else \
            geometry_for_trace(trace_path, cfg, live_frac=real_op)
        for lam in lambdas:
            run_point(cfg, out_csv, trace_name, trace_path, lam, geom, seed, streams)

    if os.path.exists(out_csv + ".raw"):
        os.remove(out_csv + ".raw")
    print(f"[robustness_sweep] wrote {out_csv}")


if __name__ == "__main__":
    main()
