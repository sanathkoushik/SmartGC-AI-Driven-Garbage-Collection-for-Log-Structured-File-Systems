"""Phase 6 - full benchmark matrix orchestrator.

Runs the Section 4 ablation matrix and writes a consolidated
``results/metrics/matrix_results.csv`` (contract v2, one row per cell, each row
carrying a ``run_id`` = hash of that cell's effective config).

Matrix
------
* the whole Phase 4a ladder (MIXED, RULE_BASED, SUP_LIKE, STAT_ML,
  LSTM_SMARTGC, LSTM_ATTN_SMARTGC) x fixed GC trigger x synthetic Zipf
* LSTM_ATTN_SMARTGC x learned GC trigger x synthetic Zipf
* LSTM_ATTN_SMARTGC x {fixed, learned} x real trace
* LSTM_ATTN_SMARTGC x {fixed, learned} x drift scenario (two concatenated
  synthetic workloads with different hot parameters)
* ``--op-sweep``: the whole ladder x fixed x synthetic at each
  ``evaluation.op_ratio_sweep`` capacity ratio -> ``results/metrics/op_sweep.csv``

Every run's effective config is hashed into ``run_id`` for reproducibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from ml.common.config import load_config, get, repo_path

PY = sys.executable
SIM = repo_path("simulator", "build", "smartgc_sim.exe")
if not os.path.exists(SIM):
    SIM = repo_path("simulator", "build", "smartgc_sim")

LADDER = ["MIXED", "RULE_BASED", "SUP_LIKE", "STAT_ML", "LSTM_SMARTGC", "LSTM_ATTN_SMARTGC"]
TRAINABLE = {"STAT_ML", "LSTM_SMARTGC", "LSTM_ATTN_SMARTGC"}
ML_POLICIES = {"RULE_BASED", "SUP_LIKE", "STAT_ML", "LSTM_SMARTGC", "LSTM_ATTN_SMARTGC"}
BEST = "LSTM_ATTN_SMARTGC"


def sh(cmd: list[str], **kw) -> None:
    print("  $ " + " ".join(str(c) for c in cmd))
    subprocess.check_call([str(c) for c in cmd], **kw)


def run_id_for(d: dict) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def default_geometry(cfg: dict) -> dict:
    return {
        "total_segments": int(get(cfg, "simulator.total_segments", 32)),
        "blocks_per_segment": int(get(cfg, "simulator.blocks_per_segment", 64)),
        "gc_threshold": int(get(cfg, "simulator.gc_free_segments_threshold", 2)),
    }


def geometry_for_trace(trace_path: str, cfg: dict, live_frac: float = 0.33) -> dict:
    """Size the storage pool from a trace's written working set so a real,
    wide-footprint trace (e.g. UMass SPC Financial1) fits with ~1/live_frac
    over-provisioning instead of overflowing the fixed synthetic geometry."""
    bps = int(get(cfg, "simulator.blocks_per_segment", 64))
    gc_thr = int(get(cfg, "simulator.gc_free_segments_threshold", 2))
    streams = int(get(cfg, "gc.migration_stream_count", 3))
    df = pd.read_csv(trace_path, usecols=["lba", "operation"])
    live = int(df.loc[df["operation"] == "W", "lba"].nunique()) or int(df["lba"].nunique())
    total_segments = max(int(get(cfg, "simulator.total_segments", 32)),
                         math.ceil(live / (live_frac * bps)) + gc_thr + streams + 4)
    return {"total_segments": total_segments, "blocks_per_segment": bps, "gc_threshold": gc_thr}


# ---------------------------------------------------------------------------
# 1. Traces
# ---------------------------------------------------------------------------
def prepare_traces(cfg: dict, seed: int) -> dict:
    raw, proc = repo_path("data", "raw"), repo_path("data", "processed")
    os.makedirs(raw, exist_ok=True)
    os.makedirs(proc, exist_ok=True)
    n = int(get(cfg, "synthetic_workload.total_requests", 5000))
    ws = int(get(cfg, "synthetic_workload.lba_range_max", 500))
    hr = float(get(cfg, "synthetic_workload.hot_ratio", 0.20))
    ht = float(get(cfg, "synthetic_workload.hot_traffic_ratio", 0.80))
    real = str(get(cfg, "evaluation.real_trace_name", "msr_cambridge_src1"))
    drift = str(get(cfg, "evaluation.drift_scenario_name", "synthetic_drift"))

    # synthetic Zipf
    sh([SIM, "--generate-only", "--requests", n, "--lba-range", ws,
        "--hot-ratio", hr, "--hot-traffic", ht, "--seed", seed, "--workload", "skewed",
        "--export-trace", os.path.join(raw, "synthetic_zipf.csv"), "--quiet"])
    sh([PY, "-m", "ml.preprocessing.normalize", "--source", "synthetic",
        "--input", os.path.join(raw, "synthetic_zipf.csv"), "--name", "synthetic_zipf"])

    # drift scenario: two halves with different hot sets/skew, concatenated
    a = os.path.join(raw, "_drift_a.csv")
    b = os.path.join(raw, "_drift_b.csv")
    sh([SIM, "--generate-only", "--requests", n // 2, "--lba-range", ws,
        "--hot-ratio", hr, "--hot-traffic", ht, "--seed", seed, "--export-trace", a, "--quiet"])
    sh([SIM, "--generate-only", "--requests", n // 2, "--lba-range", ws,
        "--hot-ratio", max(0.05, hr / 2), "--hot-traffic", min(0.95, ht + 0.1),
        "--seed", seed + 1, "--export-trace", b, "--quiet"])
    da, db = pd.read_csv(a), pd.read_csv(b)
    db["lba"] = (db["lba"] + ws // 3) % ws  # rotate the hot region
    cat = pd.concat([da, db], ignore_index=True)
    cat["timestamp"] = np.arange(len(cat))
    cat.to_csv(os.path.join(raw, f"{drift}.csv"), index=False)
    sh([PY, "-m", "ml.preprocessing.normalize", "--source", "synthetic",
        "--input", os.path.join(raw, f"{drift}.csv"), "--name", drift])

    # real trace: prefer a genuine UMass/SPC or MSR file; fall back to the stand-in.
    real_cap = int(get(cfg, "evaluation.real_trace_max_events", 150_000))
    spc_candidates = [
        os.path.join(raw, f"{real}.spc", f"{real}.spc"),
        os.path.join(raw, f"{real}.spc"),
        os.path.join(raw, "Financial1.spc", "Financial1.spc"),
        os.path.join(raw, "financial_spc", "Financial1.spc"),
    ]
    spc_raw = next((p for p in spc_candidates if os.path.exists(p)), None)
    msr_raw = os.path.join(raw, f"{real}.csv")
    if spc_raw:
        # Large OLTP trace: normalize a bounded event prefix so the LSTM +
        # simulator pipeline stays tractable (real access pattern, capped length).
        sh([PY, "-m", "ml.preprocessing.normalize", "--source", "spc",
            "--input", spc_raw, "--name", real, "--dense-lba",
            "--max-events", real_cap])
    else:
        if not os.path.exists(msr_raw):
            sh([PY, "-m", "ml.preprocessing.make_sample_trace", "--name", real])
        sh([PY, "-m", "ml.preprocessing.normalize", "--source", "msr",
            "--input", msr_raw, "--name", real, "--dense-lba"])

    traces = {
        "synthetic_zipf": repo_path("data", "processed", "synthetic_zipf.csv"),
        drift: repo_path("data", "processed", f"{drift}.csv"),
        real: repo_path("data", "processed", f"{real}.csv"),
    }
    for name, path in traces.items():
        sh([PY, "-m", "ml.preprocessing.trace_stats", "--input", path, "--name", name])
    return traces


# ---------------------------------------------------------------------------
# 2. Features + models + GC policies
# ---------------------------------------------------------------------------
def prepare_models(traces: dict, epochs: int | None, skip_train: bool) -> None:
    primary = "synthetic_zipf"
    for i, (name, path) in enumerate(traces.items()):
        args = [PY, "-m", "ml.preprocessing.features", "--input", path, "--name", name]
        if name != primary:
            args.append("--no-write-scaler")
        sh(args)

    if skip_train:
        print("  [run_matrix] --skip-train: reusing existing ml/models/<MODEL>_<ds>/")
        return
    for name in traces:
        for model in ("STAT_ML", "LSTM_SMARTGC", "LSTM_ATTN_SMARTGC"):
            cmd = [PY, "-m", "ml.training.train", "--dataset", name, "--model", model]
            if epochs is not None:
                cmd += ["--epochs", epochs]
            sh(cmd)


def prepare_gc_policies(cfg: dict, traces: dict, episodes: int | None) -> dict:
    out = {}
    for name, path in traces.items():
        synthetic = name == "synthetic_zipf"
        geom = default_geometry(cfg) if synthetic else geometry_for_trace(path, cfg)
        cmd = [PY, "-m", "ml.training.gc_controller", "--trace", path, "--name", name,
               "--total-segments", geom["total_segments"],
               "--blocks-per-segment", geom["blocks_per_segment"]]
        if not synthetic:
            # bound controller training on wide real traces (O(events x segments) in Python)
            cmd += ["--max-events", 40_000]
        if episodes is not None:
            cmd += ["--episodes", episodes]
        sh(cmd)
        out[name] = repo_path("data", "predictions", f"gc_policy_{name}.csv")
    return out


# ---------------------------------------------------------------------------
# 3. Cells
# ---------------------------------------------------------------------------
def export_predictions(policy: str, trace_path: str, trace_name: str, streams: int) -> str | None:
    if policy not in ML_POLICIES:
        return None
    stem = os.path.splitext(os.path.basename(trace_path))[0]
    name = f"{policy}_{stem}"
    cmd = [PY, "-m", "ml.inference.export", "--trace", trace_path, "--model", policy,
           "--name", name, "--streams", streams]
    if policy in TRAINABLE:
        cmd += ["--model-dir", repo_path("ml", "models", f"{policy}_{trace_name}")]
    sh(cmd)
    return repo_path("data", "predictions", f"{name}.csv")


def run_cell(cfg: dict, out_csv: str, policy: str, trace_name: str, trace_path: str,
             learned: bool, gc_policy_csv: str | None, geometry: dict, seed: int,
             streams: int, conf_thr: float) -> None:
    preds = export_predictions(policy, trace_path, trace_name, streams)
    effective = {
        "policy": policy, "trace": trace_name, "learned_trigger": learned,
        "streams": streams, "confidence_threshold": conf_thr, "seed": seed, **geometry,
    }
    rid = run_id_for(effective)
    cmd = [SIM, "--placement-policy", policy, "--trace", trace_path,
           "--total-segments", geometry["total_segments"],
           "--blocks-per-segment", geometry["blocks_per_segment"],
           "--gc-threshold", geometry["gc_threshold"],
           "--migration-streams", streams, "--seed", seed,
           "--workload-name", trace_name, "--run-id", rid,
           "--export-metrics", out_csv, "--quiet"]
    if preds:
        cmd += ["--predictions", preds, "--confidence-threshold", conf_thr]
    if learned:
        cmd += ["--learned-trigger", "--trigger-window",
                int(get(cfg, "gc.trigger_state_window_events", 500))]
        if gc_policy_csv and os.path.exists(gc_policy_csv):
            cmd += ["--gc-policy-table", gc_policy_csv]
    sh(cmd)


def matrix_cells(traces: dict, cfg: dict) -> list[dict]:
    real = str(get(cfg, "evaluation.real_trace_name", "msr_cambridge_src1"))
    drift = str(get(cfg, "evaluation.drift_scenario_name", "synthetic_drift"))
    cells = [{"policy": p, "trace": "synthetic_zipf", "learned": False} for p in LADDER]
    cells.append({"policy": BEST, "trace": "synthetic_zipf", "learned": True})
    # real trace: run the ladder baselines (MIXED / RULE_BASED) too, so the
    # before/after on real data is apples-to-apples with the same geometry.
    for p in ("MIXED", "RULE_BASED"):
        cells.append({"policy": p, "trace": real, "learned": False})
    for t in (real, drift):
        cells.append({"policy": BEST, "trace": t, "learned": False})
        cells.append({"policy": BEST, "trace": t, "learned": True})
    return cells


# ---------------------------------------------------------------------------
# 4. OP sweep (Phase 7 hook)
# ---------------------------------------------------------------------------
def op_sweep(cfg: dict, traces: dict, seed: int, streams: int, conf_thr: float) -> str:
    out_csv = repo_path("results", "metrics", "op_sweep.csv")
    if os.path.exists(out_csv):
        os.remove(out_csv)
    trace_name = "synthetic_zipf"
    trace_path = traces[trace_name]
    bps = int(get(cfg, "simulator.blocks_per_segment", 64))
    gc_thr = int(get(cfg, "simulator.gc_free_segments_threshold", 2))
    live = int(pd.read_csv(trace_path)["lba"].nunique())
    for op in get(cfg, "evaluation.op_ratio_sweep", [0.10, 0.20, 0.30]):
        total_segments = max(8, math.ceil(live / (float(op) * bps)) + gc_thr + streams + 2)
        geom = {"total_segments": total_segments, "blocks_per_segment": bps, "gc_threshold": gc_thr}
        for policy in LADDER:
            effective = {"op_ratio": op, "policy": policy, "trace": trace_name, **geom, "seed": seed}
            rid = run_id_for(effective)
            preds = export_predictions(policy, trace_path, trace_name, streams)
            cmd = [SIM, "--placement-policy", policy, "--trace", trace_path,
                   "--total-segments", total_segments, "--blocks-per-segment", bps,
                   "--gc-threshold", gc_thr, "--migration-streams", streams, "--seed", seed,
                   "--workload-name", f"{trace_name}_op{op}", "--run-id", rid,
                   "--export-metrics", out_csv, "--quiet"]
            if preds:
                cmd += ["--predictions", preds, "--confidence-threshold", conf_thr]
            sh(cmd)
    print(f"[run_matrix] OP sweep -> {out_csv}")
    return out_csv


# ---------------------------------------------------------------------------
def main() -> None:
    cfg = load_config()
    seed = int(get(cfg, "random_seed", 42))
    ap = argparse.ArgumentParser(description="SmartGC Phase 6 benchmark matrix")
    ap.add_argument("--quick", action="store_true", help="few epochs / episodes for a fast dry run")
    ap.add_argument("--skip-train", action="store_true", help="reuse existing trained models")
    ap.add_argument("--skip-prep", action="store_true", help="reuse existing traces/features/models/policies")
    ap.add_argument("--op-sweep", action="store_true", help="also run the over-provisioning sweep")
    ap.add_argument("--append", action="store_true", help="append to matrix_results.csv instead of resetting")
    args = ap.parse_args()

    epochs = 3 if args.quick else None
    episodes = 8 if args.quick else None
    streams = int(get(cfg, "gc.migration_stream_count", 3))
    conf_thr = float(get(cfg, "ml.confidence_fallback_threshold", 0.55))

    out_csv = repo_path("results", "metrics", "matrix_results.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if os.path.exists(out_csv) and not args.append:
        os.remove(out_csv)

    if args.skip_prep:
        real = str(get(cfg, "evaluation.real_trace_name", "msr_cambridge_src1"))
        drift = str(get(cfg, "evaluation.drift_scenario_name", "synthetic_drift"))
        traces = {n: repo_path("data", "processed", f"{n}.csv")
                  for n in ("synthetic_zipf", drift, real)}
        gc_policies = {n: repo_path("data", "predictions", f"gc_policy_{n}.csv") for n in traces}
    else:
        print("=== [1/4] traces ===")
        traces = prepare_traces(cfg, seed)
        print("=== [2/4] features + models ===")
        prepare_models(traces, epochs, args.skip_train)
        print("=== [3/4] GC-trigger policies ===")
        gc_policies = prepare_gc_policies(cfg, traces, episodes)

    print("=== [4/4] matrix cells ===")
    default_geom = default_geometry(cfg)
    geom_cache: dict[str, dict] = {"synthetic_zipf": default_geom}
    for cell in matrix_cells(traces, cfg):
        tname = cell["trace"]
        if tname not in geom_cache:
            # real / drift legs: size the pool from the trace's working set
            geom_cache[tname] = geometry_for_trace(traces[tname], cfg)
            print(f"  [run_matrix] geometry for {tname}: {geom_cache[tname]}")
        run_cell(cfg, out_csv, cell["policy"], tname, traces[tname],
                 cell["learned"], gc_policies.get(tname), geom_cache[tname], seed, streams, conf_thr)

    if args.op_sweep:
        print("=== [+] OP sweep ===")
        op_sweep(cfg, traces, seed, streams, conf_thr)

    df = pd.read_csv(out_csv)
    print(f"\n[run_matrix] wrote {len(df)} rows -> {out_csv}")
    cols = ["workload_name", "placement_policy", "learned_trigger_enabled", "waf",
            "valid_blocks_migrated", "gc_count", "confidence_fallback_rate"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
