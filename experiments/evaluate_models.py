#!/usr/bin/env python3
"""Evaluate every model variant and every non-learned baseline on the same data.

    python experiments/evaluate_models.py --targets msr_prxy_0 msr_proj_0

For each target workload this loads the ``scratch``, ``pretrained`` and
``pretrained_finetuned`` checkpoints, scores the workload's **held-out test
split**, and scores the baseline predictors on that identical split. Comparing
anything on different data would make the table meaningless, so the split is
built once per target and shared.

Hot/cold labels
---------------
Predicted labels use the threshold recorded in each model's metadata, which came
from that model's *training* targets. Actual labels use the same threshold
applied to the observed test intervals. The threshold is therefore never fitted
on the data it is scored against.

Writes ``results/ml/model_comparison.csv`` and
``results/ml/classification_metrics.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.config import load_config                                       # noqa: E402
from ml.evaluation.baselines import BASELINE_PREDICTORS                 # noqa: E402
from ml.evaluation.metrics import classification_metrics, regression_metrics  # noqa: E402
from ml.preprocessing.sequences import classify, hot_threshold_from_training   # noqa: E402
from ml.training.common import (                                        # noqa: E402
    DEFAULT_MAX_SEQUENCES_PER_TRACE,
    MODELS_DIR,
    build_split_pool,
    load_checkpoint,
    predict_intervals,
    resolve_device,
)
from ml.training.finetune import MODEL_TYPES, default_output_path       # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "ml"
COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"
CLASSIFICATION_PATH = RESULTS_DIR / "classification_metrics.csv"

COMPARISON_COLUMNS = [
    "trace", "predictor", "family", "sequence_length", "test_samples",
    "mae_us", "rmse_us", "median_abs_error_us", "mae_log1p",
    "hot_threshold_us", "accuracy", "precision", "recall", "f1", "roc_auc",
    "actual_hot_fraction", "predicted_hot_fraction",
]
CLASSIFICATION_COLUMNS = [
    "trace", "predictor", "family", "hot_threshold_us", "n",
    "accuracy", "precision", "recall", "f1", "roc_auc",
    "true_positives", "false_positives", "true_negatives", "false_negatives",
    "actual_hot_fraction", "predicted_hot_fraction",
]


def evaluate_target(trace_id: str,
                    config_path: Path | None = None,
                    max_sequences: int | None = DEFAULT_MAX_SEQUENCES_PER_TRACE,
                    device_preference: str = "auto",
                    verbose: bool = True) -> list[dict[str, Any]]:
    """Score every available model variant and baseline on one target's test split."""
    config = load_config(config_path)
    ml_config = config.ml
    device = resolve_device(device_preference)
    rows: list[dict[str, Any]] = []

    # Built once at the config's sequence length; a model trained at a different
    # length gets its own pool below, because the window size changes the data.
    pools: dict[int, Any] = {}

    def pool_for(sequence_length: int):
        if sequence_length not in pools:
            pools[sequence_length] = build_split_pool(
                [trace_id], ml_config, max_sequences=max_sequences,
                sequence_length=sequence_length)
        return pools[sequence_length]

    for model_type in MODEL_TYPES:
        checkpoint = default_output_path(model_type, trace_id)
        if not checkpoint.is_file():
            if verbose:
                print(f"  {model_type}: checkpoint missing ({checkpoint.name}), skipping")
            continue

        model, scaler, metadata = load_checkpoint(checkpoint, device)
        sequence_length = model.hyperparameters.sequence_length
        pool = pool_for(sequence_length)
        if pool.test["targets"].size == 0:
            if verbose:
                print(f"  {model_type}: empty test split, skipping")
            continue

        threshold = float(metadata.get("hot_threshold_microseconds", 0.0))
        if threshold <= 0.0:
            raise ValueError(f"{checkpoint.name} has no training-derived hot threshold")

        predicted = predict_intervals(model, pool.test["inputs"], scaler, device)
        regression = regression_metrics(pool.test["targets"], predicted)
        actual_labels = classify(pool.test["targets"], threshold)
        predicted_labels = classify(predicted, threshold)
        # A shorter predicted interval means hotter, so the ranking score is the
        # negated prediction.
        classification = classification_metrics(actual_labels, predicted_labels,
                                                scores=-predicted)

        rows.append({
            "trace": trace_id, "predictor": model_type, "family": "lstm",
            "sequence_length": sequence_length,
            "test_samples": regression.n,
            "mae_us": regression.mae, "rmse_us": regression.rmse,
            "median_abs_error_us": regression.median_absolute_error,
            "mae_log1p": regression.mae_log1p,
            "hot_threshold_us": threshold,
            **{key: getattr(classification, key) for key in
               ("accuracy", "precision", "recall", "f1", "roc_auc",
                "actual_hot_fraction", "predicted_hot_fraction")},
            "_confusion": classification.to_dict(),
        })
        if verbose:
            print(f"  {model_type:<22} MAE {regression.mae:>14,.0f} us   "
                  f"F1 {classification.f1:.4f}   acc {classification.accuracy:.4f}")

    # Baselines are scored on the pool at the configured sequence length, using
    # a threshold derived from that pool's own TRAINING targets - the same rule
    # the models followed, so no baseline gets an unfair look at the test set.
    pool = pool_for(ml_config.sequence_length)
    if pool.test["targets"].size and pool.train["targets"].size:
        baseline_threshold = hot_threshold_from_training(pool.train["targets"],
                                                         ml_config.hot_percentile_cutoff)
        actual_labels = classify(pool.test["targets"], baseline_threshold)
        for name, predictor in BASELINE_PREDICTORS.items():
            predicted = predictor(pool.test["inputs"])
            regression = regression_metrics(pool.test["targets"], predicted)
            predicted_labels = classify(predicted, baseline_threshold)
            classification = classification_metrics(actual_labels, predicted_labels,
                                                    scores=-predicted)
            rows.append({
                "trace": trace_id, "predictor": name, "family": "baseline",
                "sequence_length": ml_config.sequence_length,
                "test_samples": regression.n,
                "mae_us": regression.mae, "rmse_us": regression.rmse,
                "median_abs_error_us": regression.median_absolute_error,
                "mae_log1p": regression.mae_log1p,
                "hot_threshold_us": baseline_threshold,
                **{key: getattr(classification, key) for key in
                   ("accuracy", "precision", "recall", "f1", "roc_auc",
                    "actual_hot_fraction", "predicted_hot_fraction")},
                "_confusion": classification.to_dict(),
            })
            if verbose:
                print(f"  {name:<22} MAE {regression.mae:>14,.0f} us   "
                      f"F1 {classification.f1:.4f}   acc {classification.accuracy:.4f}")

    return rows


def write_results(rows: Sequence[dict[str, Any]],
                  comparison_path: Path = COMPARISON_PATH,
                  classification_path: Path = CLASSIFICATION_PATH) -> None:
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {comparison_path.relative_to(REPO_ROOT)}  ({len(rows)} rows)")

    with classification_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLASSIFICATION_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            confusion = row.get("_confusion", {})
            writer.writerow({**row, **confusion, "n": confusion.get("n", row.get("test_samples"))})
    print(f"Wrote {classification_path.relative_to(REPO_ROOT)}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--max-sequences", type=int, default=DEFAULT_MAX_SEQUENCES_PER_TRACE)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows: list[dict[str, Any]] = []
    for trace_id in args.targets:
        print(f"\nEvaluating {trace_id} on its held-out test split")
        rows.extend(evaluate_target(trace_id, config_path=args.config,
                                    max_sequences=args.max_sequences,
                                    device_preference=args.device))
    if not rows:
        print("No models could be evaluated.", file=sys.stderr)
        return 1
    write_results(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
