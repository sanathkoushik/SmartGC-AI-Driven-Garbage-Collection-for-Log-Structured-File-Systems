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
from ml.inference.robust_blend import blend_interval

HYBRID_POLICY = "HYBRID_ROBUST_SMARTGC"
HYBRID_BASE_MODEL = "LSTM_ATTN_SMARTGC"


def _load_scaler(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _build_or_load_sequences(trace_path: str, seq_len: int, rolling_window: int, scaler: dict):
    """Build the per-write-event inference windows for a trace, caching the
    (expensive on a 400k-event real trace) result next to the trace keyed on the
    trace + scaler mtimes and seq_len. Every ML policy on the same trace then
    reuses one build instead of recomputing it."""
    cache = os.path.splitext(trace_path)[0] + f".infer_seq_sl{seq_len}.npz"
    scaler_path = repo_path("ml", "models", "scaler.json")
    key = f"{os.path.getmtime(trace_path):.0f}|{os.path.getmtime(scaler_path):.0f}|{seq_len}"
    if os.path.exists(cache):
        try:
            z = np.load(cache, allow_pickle=True)
            if str(z["key"]) == key:
                return z["Xs"], z["meta"], z["has_history"]
        except Exception:
            pass
    df = pd.read_csv(trace_path)
    df.columns = [c.strip().lower() for c in df.columns]
    feat = build_event_features(df, rolling_window)
    X, meta, has_history = build_inference_sequences(feat, seq_len)
    Xs = apply_scaler(X, scaler)
    try:
        np.savez_compressed(cache, Xs=Xs, meta=meta, has_history=has_history, key=np.array(key))
    except Exception:
        pass
    return Xs, meta, has_history


def run_export(trace_path: str, policy: str, model_dir: str | None, name: str,
               n_streams: int, cfg: dict, robust_lambda: float | None = None) -> dict:
    seq_len = int(get(cfg, "ml.sequence_length", 10))
    rolling_window = seq_len
    scaler = _load_scaler(repo_path("ml", "models", "scaler.json"))

    Xs, meta, has_history = _build_or_load_sequences(trace_path, seq_len, rolling_window, scaler)
    lbas, widx, ts = meta[:, 0], meta[:, 1], meta[:, 2]

    # Model + always-available RULE_BASED fallback predictions. HYBRID_ROBUST_SMARTGC
    # (Phase 8) reuses the LSTM_ATTN_SMARTGC checkpoint verbatim (see registry.py) --
    # only the confidence handling below differs for that policy.
    is_hybrid = policy == HYBRID_POLICY
    build_name = HYBRID_BASE_MODEL if is_hybrid else policy
    model = build(build_name, scaler, cfg)
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

    lam = float(get(cfg, "ml.robustness_lambda", 0.35)) if robust_lambda is None else float(robust_lambda)
    blended_interval, trust_w = blend_interval(model_interval, rb_interval, confidence, lam)

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
        if is_hybrid:
            # Continuous, confidence-weighted, robustness-capped blend
            # (ml/inference/robust_blend.py) replaces the binary gate below --
            # this generalises ConfidenceGate's hard cutoff into a tunable
            # consistency/robustness tradeoff (Lange/Naor/Yadgar, SIGMETRICS'25
            # framing; see docs/related_work.md).
            interval = float(blended_interval[i])
        else:
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
        "confidence_fallback_rate": round(gate.fallback_rate, 4) if not is_hybrid else None,
        "drift_needs_retrain": dd.needs_retrain,
        "drift_last_flag_event": dd.last_flag_event,
        "drift_status_json": drift_path,
    }
    if is_hybrid:
        summary["robustness_lambda"] = lam
        summary["mean_trust_weight"] = round(float(np.mean(trust_w)), 4)
    return summary


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="SmartGC Phase 4b predictions exporter")
    ap.add_argument("--trace", required=True, help="normalized trace CSV")
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--model-dir", default=None, help="trained artefact dir (ml/models/<MODEL>_<ds>)")
    ap.add_argument("--name", default=None, help="predictions name (default: <model>_<trace stem>)")
    ap.add_argument("--streams", type=int, default=int(get(cfg, "gc.migration_stream_count", 3)))
    ap.add_argument("--robust-lambda", type=float, default=None,
                    help="HYBRID_ROBUST_SMARTGC only: override ml.robustness_lambda for this run")
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.trace))[0]
    name = args.name or f"{args.model}_{stem}"
    if args.model_dir is None and args.model in ("STAT_ML", "LSTM_SMARTGC", "LSTM_ATTN_SMARTGC"):
        args.model_dir = repo_path("ml", "models", f"{args.model}_{stem}")
    elif args.model_dir is None and args.model == HYBRID_POLICY:
        # Reuses the LSTM_ATTN_SMARTGC checkpoint verbatim -- no separate training.
        args.model_dir = repo_path("ml", "models", f"{HYBRID_BASE_MODEL}_{stem}")

    summary = run_export(args.trace, args.model, args.model_dir, name, max(1, args.streams),
                          cfg, robust_lambda=args.robust_lambda)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
