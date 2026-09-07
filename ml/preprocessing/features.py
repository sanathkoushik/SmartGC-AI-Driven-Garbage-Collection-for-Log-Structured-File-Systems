"""Phase 3 - multivariate feature engineering & sequence construction.

For every write event we build the per-LBA feature vector

    [time_since_last_write, rolling_mean_interval, rolling_std_interval,
     write_count_so_far, request_size]

and assemble, per LBA, left-padded sequences of ``sequence_length`` such
vectors.  The regression target is that LBA's *next* rewrite interval (in write
events).  The chronological 70/15/15 split and the fixed seed come from
``config.yaml``.  The fitted normalization parameters are written to
``ml/models/scaler.json`` so inference reproduces them exactly (never refit at
inference time).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from ml.common.config import load_config, get, repo_path, seed as cfg_seed
from ml.common.io_contracts import CONTRACT_VERSION

FEATURES = [
    "time_since_last_write",
    "rolling_mean_interval",
    "rolling_std_interval",
    "write_count_so_far",
    "request_size",
]
SCALER_PATH = repo_path("ml", "models", "scaler.json")


# ---------------------------------------------------------------------------
# Per-event feature table
# ---------------------------------------------------------------------------
def build_event_features(df: pd.DataFrame, rolling_window: int) -> pd.DataFrame:
    """One row per write event, in chronological order, with the 5 features and
    the regression target ``target_interval`` (NaN for each LBA's final write)."""
    w = df[df["operation"] == "W"].sort_values("timestamp", kind="stable").reset_index(drop=True)
    w["widx"] = np.arange(len(w), dtype=np.int64)

    g = w.groupby("lba", sort=False)
    prev_widx = g["widx"].shift(1)
    w["time_since_last_write"] = (w["widx"] - prev_widx).fillna(0.0)
    w["write_count_so_far"] = g.cumcount().astype(np.float64) + 1.0
    w["request_size"] = w["size"].astype(np.float64)

    # Rolling stats over each LBA's *past* observed intervals (exclude current).
    def _roll(series: pd.Series, fn: str) -> pd.Series:
        shifted = series.shift(1)
        r = shifted.rolling(window=rolling_window, min_periods=1)
        return getattr(r, fn)()

    w["rolling_mean_interval"] = (
        g["time_since_last_write"].transform(lambda s: _roll(s, "mean")).fillna(0.0)
    )
    w["rolling_std_interval"] = (
        g["time_since_last_write"].transform(lambda s: _roll(s, "std")).fillna(0.0)
    )

    # Target: gap to this LBA's next write.
    next_widx = g["widx"].shift(-1)
    w["target_interval"] = (next_widx - w["widx"])
    return w


# ---------------------------------------------------------------------------
# Sequence assembly
# ---------------------------------------------------------------------------
def build_sequences(feat: pd.DataFrame, seq_len: int):
    """Return (X, y, meta) where X is [N, seq_len, F], y is [N] next-interval
    targets, and meta carries (lba, widx, timestamp) per sample for aligning
    predictions back onto trace events."""
    fvals = feat[FEATURES].to_numpy(np.float64)
    lba = feat["lba"].to_numpy(np.int64)
    widx = feat["widx"].to_numpy(np.int64)
    ts = feat["timestamp"].to_numpy(np.int64)
    target = feat["target_interval"].to_numpy(np.float64)

    # Per-LBA row indices in chronological order.
    order_by_lba: dict[int, list[int]] = {}
    for i, l in enumerate(lba):
        order_by_lba.setdefault(int(l), []).append(i)

    X, y, meta = [], [], []
    F = len(FEATURES)
    for rows in order_by_lba.values():
        for pos in range(len(rows)):
            gi = rows[pos]
            if not np.isfinite(target[gi]):
                continue  # LBA's final write - no supervised target
            if pos < 1:
                continue  # need at least one prior write for a real step
            hist = rows[max(0, pos - seq_len + 1): pos + 1]
            seq = np.zeros((seq_len, F), dtype=np.float64)
            seq[seq_len - len(hist):] = fvals[hist]
            X.append(seq)
            y.append(target[gi])
            meta.append((int(lba[gi]), int(widx[gi]), int(ts[gi])))

    if not X:
        return (np.zeros((0, seq_len, F)), np.zeros((0,)),
                np.zeros((0, 3), np.int64))
    order = np.argsort([m[1] for m in meta])  # chronological by target widx
    X = np.asarray(X)[order]
    y = np.asarray(y)[order]
    meta = np.asarray(meta, dtype=np.int64)[order]
    return X, y, meta


def chronological_split(n: int, tr: float, va: float):
    i_tr = int(round(tr * n))
    i_va = int(round((tr + va) * n))
    return slice(0, i_tr), slice(i_tr, i_va), slice(i_va, n)


# ---------------------------------------------------------------------------
# Scaler (transparent, JSON-serializable - no sklearn dependency here)
# ---------------------------------------------------------------------------
def _real_rows(X: np.ndarray) -> np.ndarray:
    """Mask of timesteps that are not left-padding (any non-zero feature)."""
    flat = X.reshape(-1, X.shape[-1])
    return flat, np.any(flat != 0.0, axis=1)


def fit_scaler(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    if X_train.size:
        flat, real = _real_rows(X_train)
        flat = flat[real] if real.any() else flat
    else:
        flat = np.zeros((1, len(FEATURES)))
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-8] = 1.0
    y_log = np.log1p(np.clip(y_train, 0, None)) if y_train.size else np.zeros(1)
    return {
        "contract_version": CONTRACT_VERSION,
        "features": FEATURES,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "target_transform": "log1p",
        "target_mean": float(y_log.mean()),
        "target_std": float(y_log.std() if y_log.std() > 1e-8 else 1.0),
    }


def apply_scaler(X: np.ndarray, scaler: dict) -> np.ndarray:
    mean = np.asarray(scaler["feature_mean"])
    std = np.asarray(scaler["feature_std"])
    if X.size == 0:
        return X
    pad_mask = ~np.any(X != 0.0, axis=-1)  # [N, seq_len] left-padding positions
    out = (X - mean) / std
    out[pad_mask] = 0.0  # keep padding neutral after scaling
    return out


def transform_target(y: np.ndarray, scaler: dict) -> np.ndarray:
    yl = np.log1p(np.clip(y, 0, None))
    return (yl - scaler["target_mean"]) / scaler["target_std"]


def inverse_target(y_scaled: np.ndarray, scaler: dict) -> np.ndarray:
    yl = y_scaled * scaler["target_std"] + scaler["target_mean"]
    return np.expm1(yl)


# ---------------------------------------------------------------------------
def main() -> None:
    cfg = load_config()
    seq_len = int(get(cfg, "ml.sequence_length", 10))
    tr = float(get(cfg, "ml.train_split", 0.70))
    va = float(get(cfg, "ml.val_split", 0.15))
    rolling_window = int(get(cfg, "ml.sequence_length", 10))

    ap = argparse.ArgumentParser(description="SmartGC Phase 3 feature/sequence builder")
    ap.add_argument("--input", required=True, help="normalized trace CSV")
    ap.add_argument("--name", help="dataset name (default: input stem)")
    ap.add_argument("--out", help="output .npz (default: data/processed/sequences_<name>.npz)")
    ap.add_argument("--scaler-out", default=SCALER_PATH)
    ap.add_argument("--no-write-scaler", action="store_true",
                    help="do not overwrite ml/models/scaler.json (use for aux datasets)")
    args = ap.parse_args()

    np.random.seed(cfg_seed(cfg))
    name = args.name or os.path.splitext(os.path.basename(args.input))[0]
    out = args.out or repo_path("data", "processed", f"sequences_{name}.npz")

    df = pd.read_csv(args.input)
    df.columns = [c.strip().lower() for c in df.columns]
    feat = build_event_features(df, rolling_window)
    X, y, meta = build_sequences(feat, seq_len)
    if len(X) == 0:
        raise SystemExit("features: no supervised sequences could be built (trace has no rewrites?)")

    s_tr, s_va, s_te = chronological_split(len(X), tr, va)
    scaler = fit_scaler(X[s_tr], y[s_tr])
    if not args.no_write_scaler:
        os.makedirs(os.path.dirname(args.scaler_out), exist_ok=True)
        with open(args.scaler_out, "w", encoding="utf-8") as fh:
            json.dump(scaler, fh, indent=2)

    Xs = apply_scaler(X, scaler)
    ys = transform_target(y, scaler)

    np.savez_compressed(
        out,
        X_train=Xs[s_tr], y_train=ys[s_tr], meta_train=meta[s_tr],
        X_val=Xs[s_va], y_val=ys[s_va], meta_val=meta[s_va],
        X_test=Xs[s_te], y_test=ys[s_te], meta_test=meta[s_te],
        y_raw_train=y[s_tr], y_raw_val=y[s_va], y_raw_test=y[s_te],
        feature_names=np.array(FEATURES),
        seq_len=np.array([seq_len]),
    )
    print(f"[features] {name}: {len(X):,} sequences  "
          f"(train {s_tr.stop:,} / val {s_va.stop - s_va.start:,} / test {len(X) - s_te.start:,})")
    print(f"[features] X shape={Xs.shape}  target median(raw)={np.median(y):.1f}")
    print(f"[features] wrote {out}")
    if not args.no_write_scaler:
        print(f"[features] wrote {args.scaler_out}")


if __name__ == "__main__":
    main()
