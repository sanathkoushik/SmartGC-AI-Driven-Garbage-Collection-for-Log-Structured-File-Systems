"""Phase 4b - assemble predictions.csv (contract v2) for the C++ simulator.

Pipeline: normalized trace -> per-event features -> one inference window per
write event -> model prediction (+ MC-dropout confidence for the LSTM rungs) ->
confidence gate (fall back to RULE_BASED per block) -> rolling-percentile
HOT/COLD + multi-stream bucket -> drift detector -> one CSV row per write event,
in lockstep with the simulator's replay.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from ml.common.config import load_config, get, repo_path
from ml.common.io_contracts import PREDICTION_COLUMNS, STREAM_NAMES
from ml.models.registry import build, REGISTRY
from ml.models.rule_based import RuleBasedModel
from ml.preprocessing.features import (
    build_event_features, build_inference_sequences, apply_scaler,
)
from ml.inference.rolling_cutoff import RollingPercentileCutoff
from ml.inference.drift import DriftDetector
from ml.inference.confidence_gate import ConfidenceGate, mc_dropout_confidence


def _load_scaler(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def run_export(trace_path: str, policy: str, model_dir: str | None, name: str,
               n_streams: int, cfg: dict) -> dict:
    ml = cfg.get("ml", {})
    seq_len = int(get(cfg, "ml.sequence_length", 10))
    rolling_window = seq_len
    scaler = _load_scaler(repo_path("ml", "models", "scaler.json"))

    df = pd.read_csv(trace_path)
    df.columns = [c.strip().lower() for c in df.columns]
    feat = build_event_features(df, rolling_window)
    X, meta, has_history = build_inference_sequences(feat, seq_len)
    Xs = apply_scaler(X, scaler)
    lbas, widx, ts = meta[:, 0], meta[:, 1], meta[:, 2]

    # Model + always-available RULE_BASED fallback predictions.
    model = build(policy, scaler, cfg)
    if model.trainable:
        if not model_dir or not os.path.isdir(model_dir):
            raise SystemExit(f"export: trained model dir required for {policy}: {model_dir}")
        model.load(model_dir)
    rb = RuleBasedModel(scaler, cfg)
    rb_interval = rb.predict_interval(Xs)

    k = int(get(cfg, "ml.confidence_ensemble_size", 3))
    model_interval, confidence = mc_dropout_confidence(model, Xs, k)
    # First-ever write of an LBA has no real history -> distrust it.
    confidence = np.where(has_history, confidence, np.minimum(confidence, 0.30))
    model_interval = np.where(has_history, model_interval, rb_interval)

    gate = ConfidenceGate(float(get(cfg, "ml.confidence_fallback_threshold", 0.55)))
    rc = RollingPercentileCutoff(
        window=int(get(cfg, "ml.hot_cutoff_window_events", 2000)),
        hot_percentile=float(get(cfg, "ml.hot_percentile_cutoff", 30)),
    )
    dd = DriftDetector(
        window=int(get(cfg, "ml.drift_window_events", 2000)),
        kl_threshold=float(get(cfg, "ml.drift_kl_threshold", 0.15)),
        name=name,
    )

    rows = []
    for i in range(len(lbas)):
        trust = gate.decide(float(confidence[i]))
        interval = float(model_interval[i] if trust else rb_interval[i])
        interval = max(interval, 0.0)

        cls = rc.classify(interval)                     # decide before recording
        bucket = rc.stream_class(interval, n_streams)
        rc.update(interval)
        dd.update(interval)

        rows.append((int(ts[i]), int(lbas[i]), round(interval, 4),
                     "HOT" if cls == 1 else "COLD", int(bucket),
                     round(float(confidence[i]), 4)))
    dd.finalize()

    out_csv = repo_path("data", "predictions", f"{name}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(rows, columns=PREDICTION_COLUMNS).to_csv(out_csv, index=False)

    drift_path = repo_path("results", "metrics", f"drift_status_{name}.json")
    dd.write_status(drift_path)

    hot = sum(1 for r in rows if r[3] == "HOT")
    buckets = np.bincount([r[4] for r in rows], minlength=n_streams).tolist()
    summary = {
        "predictions_csv": out_csv,
        "rows": len(rows),
        "policy": policy,
        "hot_fraction": round(hot / len(rows), 4) if rows else 0.0,
        "stream_bucket_counts": {STREAM_NAMES.get(i, str(i)): int(c) for i, c in enumerate(buckets)},
        "confidence_fallback_rate": round(gate.fallback_rate, 4),
        "drift_needs_retrain": dd.needs_retrain,
        "drift_last_flag_event": dd.last_flag_event,
        "drift_status_json": drift_path,
    }
    return summary


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="SmartGC Phase 4b predictions exporter")
    ap.add_argument("--trace", required=True, help="normalized trace CSV")
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--model-dir", default=None, help="trained artefact dir (ml/models/<MODEL>_<ds>)")
    ap.add_argument("--name", default=None, help="predictions name (default: <model>_<trace stem>)")
    ap.add_argument("--streams", type=int, default=int(get(cfg, "gc.migration_stream_count", 3)))
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.trace))[0]
    name = args.name or f"{args.model}_{stem}"
    if args.model_dir is None and args.model in ("STAT_ML", "LSTM_SMARTGC", "LSTM_ATTN_SMARTGC"):
        args.model_dir = repo_path("ml", "models", f"{args.model}_{stem}")

    summary = run_export(args.trace, args.model, args.model_dir, name, max(1, args.streams), cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
