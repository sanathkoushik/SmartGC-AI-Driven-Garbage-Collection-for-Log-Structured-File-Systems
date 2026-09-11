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

LEARNERS = ["STAT_ML", "LSTM_SMARTGC", "LSTM_ATTN_SMARTGC", "HYBRID_ROBUST_SMARTGC"]
# HYBRID_ROBUST_SMARTGC (Phase 8) reuses the LSTM_ATTN_SMARTGC checkpoint verbatim
# (ml/models/registry.py) -- report its cost from that artefact dir; the extra
# robustness blend at inference is an O(1) elementwise op per event, negligible
# next to the LSTM forward pass already being measured.
CHECKPOINT_ALIAS = {"HYBRID_ROBUST_SMARTGC": "LSTM_ATTN_SMARTGC"}


def _artifact_bytes(d: str) -> int:
    total = 0
    for root, _, files in os.walk(d):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def _waf_lookup(matrix_csv: str, workload: str) -> dict:
    if not os.path.exists(matrix_csv):
        return {}
    df = pd.read_csv(matrix_csv)
    df = df[(df["workload_name"] == workload) & (df["learned_trigger_enabled"] == 0)]
    return dict(zip(df["placement_policy"], df["waf"]))


def _cost_rows(dataset: str, cfg: dict, scaler: dict, batch_size: int) -> list[dict]:
    npz_path = repo_path("data", "processed", f"sequences_{dataset}.npz")
    if not os.path.exists(npz_path):
        print(f"[cost] skip dataset {dataset}: {npz_path} missing")
        return []
    Xte = np.load(npz_path, allow_pickle=True)["X_test"]
    waf = _waf_lookup(repo_path("results", "metrics", "matrix_results.csv"), dataset)

    rows = []
    for name in LEARNERS:
        checkpoint_owner = CHECKPOINT_ALIAS.get(name, name)
        mdir = repo_path("ml", "models", f"{checkpoint_owner}_{dataset}")
        if not os.path.isdir(mdir):
            print(f"[cost] skip {name}/{dataset}: no trained artefact")
            continue
        model = build(checkpoint_owner, scaler, cfg).load(mdir)
        model.predict_interval(Xte[: min(len(Xte), 128)])  # warm up
        tracemalloc.start()
        lat = model.measure_latency(Xte, batch_size=batch_size, repeats=5)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append({
            "model": name,
            "dataset": dataset,
            "param_count": model.param_count(),
            "latency_ms_per_batch": round(lat["latency_ms_per_batch"], 4),
            "throughput_pred_per_s": round(lat["throughput_pred_per_s"], 1),
            "peak_infer_mem_mb": round(peak / 1024 / 1024, 3),
            "artifact_bytes": _artifact_bytes(mdir),
            "waf": round(float(waf.get(name, float("nan"))), 6),
        })
        print(f"[cost] {name}/{dataset}: params={rows[-1]['param_count']:,}  "
              f"{rows[-1]['latency_ms_per_batch']} ms/batch  "
              f"{rows[-1]['throughput_pred_per_s']:,} pred/s  WAF={rows[-1]['waf']}")
    return rows


def main() -> None:
    cfg = load_config()
    real = str(get(cfg, "evaluation.real_trace_name", "financial1"))
    ap = argparse.ArgumentParser(description="SmartGC Phase 7 model cost report")
    ap.add_argument("--datasets", default=f"synthetic_zipf,{real}",
                    help="comma-separated dataset names")
    ap.add_argument("--batch-size", type=int, default=int(get(cfg, "ml.batch_size", 64)))
    ap.add_argument("--out", default=repo_path("results", "metrics", "model_cost.csv"))
    args = ap.parse_args()

    scaler = json.load(open(repo_path("ml", "models", "scaler.json"), encoding="utf-8"))
    rows = []
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        rows += _cost_rows(ds, cfg, scaler, args.batch_size)

    if not rows:
        raise SystemExit("[cost] no trained models found - run ml.training.train first")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"[cost] wrote {args.out}")


if __name__ == "__main__":
    main()
