"""Phase 4a - train one rung of the baseline ladder under a common protocol.

Loads a Phase 3 sequence dataset (``data/processed/sequences_<name>.npz``) plus
the shared ``ml/models/scaler.json``, fits the requested model, reports raw
next-rewrite-interval MAE/RMSE on val + test, and saves the fitted artefact to
``ml/models/<MODEL>_<dataset>/``.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from ml.common.config import load_config, repo_path
from ml.models.registry import build, REGISTRY
from ml.preprocessing.features import inverse_target


def _resolve_dataset(arg: str) -> str:
    if os.path.isfile(arg):
        return arg
    cand = repo_path("data", "processed", f"sequences_{arg}.npz")
    if os.path.isfile(cand):
        return cand
    raise SystemExit(f"train: dataset not found: {arg}")


def _metrics(pred_raw: np.ndarray, y_raw: np.ndarray) -> dict:
    if len(y_raw) == 0:
        return {"mae": 0.0, "rmse": 0.0, "n": 0}
    err = pred_raw - y_raw
    return {"mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "n": int(len(y_raw))}


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="SmartGC Phase 4a model trainer")
    ap.add_argument("--dataset", required=True, help="npz path or dataset name")
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--scaler", default=repo_path("ml", "models", "scaler.json"))
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    ds_path = _resolve_dataset(args.dataset)
    ds_name = os.path.splitext(os.path.basename(ds_path))[0].replace("sequences_", "")
    with open(args.scaler, "r", encoding="utf-8") as fh:
        scaler = json.load(fh)
    data = np.load(ds_path, allow_pickle=True)

    Xtr, ytr = data["X_train"], data["y_train"]
    Xva, yva = data["X_val"], data["y_val"]
    Xte, yte = data["X_test"], data["y_test"]
    yva_raw, yte_raw = data["y_raw_val"], data["y_raw_test"]

    model = build(args.model, scaler, cfg)
    print(f"[train] model={args.model} dataset={ds_name} "
          f"train={len(Xtr):,} val={len(Xva):,} test={len(Xte):,} trainable={model.trainable}")
    model.fit(Xtr, ytr, Xva, yva)

    val_m = _metrics(model.predict_interval(Xva), yva_raw)
    test_m = _metrics(model.predict_interval(Xte), yte_raw)
    # Naive reference: predict the training-set median raw interval everywhere.
    ref = float(np.median(inverse_target(ytr, scaler))) if len(ytr) else 0.0
    ref_m = _metrics(np.full_like(yte_raw, ref, dtype=float), yte_raw)

    out_dir = args.out_dir or repo_path("ml", "models", f"{args.model}_{ds_name}")
    model.save(out_dir)
    report = {
        "model": args.model, "dataset": ds_name,
        "param_count": model.param_count(),
        "val": val_m, "test": test_m, "naive_median_test": ref_m,
    }
    with open(os.path.join(out_dir, "train_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"[train] params={report['param_count']:,}")
    print(f"[train] val  MAE={val_m['mae']:.2f}  RMSE={val_m['rmse']:.2f}")
    print(f"[train] test MAE={test_m['mae']:.2f}  RMSE={test_m['rmse']:.2f}  "
          f"(naive-median MAE={ref_m['mae']:.2f})")
    print(f"[train] saved -> {out_dir}")


if __name__ == "__main__":
    main()
