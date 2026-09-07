"""Phase 7 - model cost report.

For each learning rung (STAT_ML, LSTM_SMARTGC, LSTM_ATTN_SMARTGC) reports
parameter count, average inference latency per prediction batch, throughput,
peak Python-allocated memory during inference, and on-disk artefact size - so
the writeup can put WAF gains next to their compute cost (the Shiro-style
host-side-ML-overhead critique).  WAF for each policy on the synthetic Zipf
workload is pulled from ``matrix_results.csv`` when present.
"""
from __future__ import annotations

import argparse
import json
import os
import tracemalloc

import numpy as np
import pandas as pd

from ml.common.config import load_config, get, repo_path
from ml.models.registry import build

LEARNERS = ["STAT_ML", "LSTM_SMARTGC", "LSTM_ATTN_SMARTGC"]


def _artifact_bytes(d: str) -> int:
    total = 0
    for root, _, files in os.walk(d):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def _waf_lookup(matrix_csv: str) -> dict:
    if not os.path.exists(matrix_csv):
        return {}
    df = pd.read_csv(matrix_csv)
    df = df[(df["workload_name"] == "synthetic_zipf") & (df["learned_trigger_enabled"] == 0)]
    return dict(zip(df["placement_policy"], df["waf"]))


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="SmartGC Phase 7 model cost report")
    ap.add_argument("--dataset", default="synthetic_zipf")
    ap.add_argument("--batch-size", type=int, default=int(get(cfg, "ml.batch_size", 64)))
    ap.add_argument("--out", default=repo_path("results", "metrics", "model_cost.csv"))
    args = ap.parse_args()

    scaler = json.load(open(repo_path("ml", "models", "scaler.json"), encoding="utf-8"))
    npz = np.load(repo_path("data", "processed", f"sequences_{args.dataset}.npz"), allow_pickle=True)
    Xte = npz["X_test"]
    waf = _waf_lookup(repo_path("results", "metrics", "matrix_results.csv"))

    rows = []
    for name in LEARNERS:
        mdir = repo_path("ml", "models", f"{name}_{args.dataset}")
        if not os.path.isdir(mdir):
            print(f"[cost] skip {name}: no trained artefact at {mdir}")
            continue
        model = build(name, scaler, cfg).load(mdir)

        model.predict_interval(Xte[: min(len(Xte), 128)])  # warm up
        tracemalloc.start()
        lat = model.measure_latency(Xte, batch_size=args.batch_size, repeats=5)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        rows.append({
            "model": name,
            "dataset": args.dataset,
            "param_count": model.param_count(),
            "latency_ms_per_batch": round(lat["latency_ms_per_batch"], 4),
            "throughput_pred_per_s": round(lat["throughput_pred_per_s"], 1),
            "peak_infer_mem_mb": round(peak / 1024 / 1024, 3),
            "artifact_bytes": _artifact_bytes(mdir),
            "waf_synthetic_zipf": round(float(waf.get(name, float("nan"))), 6),
        })
        print(f"[cost] {name}: params={rows[-1]['param_count']:,}  "
              f"{rows[-1]['latency_ms_per_batch']} ms/batch  "
              f"{rows[-1]['throughput_pred_per_s']:,} pred/s  "
              f"WAF={rows[-1]['waf_synthetic_zipf']}")

    if not rows:
        raise SystemExit("[cost] no trained models found - run ml.training.train first")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"[cost] wrote {args.out}")


if __name__ == "__main__":
    main()
