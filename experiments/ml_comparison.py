#!/usr/bin/env python3
"""Classical ML models vs. the existing baselines and LSTMs, on one identical split.

    python experiments/ml_comparison.py --targets msr_rsrch_0 msr_src2_0 systor_lun0

For every target this trains linear regression, a random forest and XGBoost on the
target's own chronological training split (XGBoost early-stops on the validation
split, as the LSTM does), then scores them on the same held-out test split, at the
same sequence length, with the same hot/cold threshold rule as
``experiments/transfer_comparison.py``.

Writes
    results/ml/classical_ml_comparison.csv         the new models only
    results/final/full_model_comparison.csv        baselines + LSTMs + classical ML
    models/classical/<model>__<trace>.joblib/.json fitted models and metadata
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.transfer_comparison import COLUMNS, _model_sequence_length, _row  # noqa: E402
from ml.config import environment_metadata, load_config                          # noqa: E402
from ml.evaluation.metrics import classification_metrics, regression_metrics     # noqa: E402
from ml.inference.predict_classical import classical_model_path                  # noqa: E402
from ml.models.classical import CLASSICAL_MODEL_TYPES, DISPLAY_NAME, fit_classical  # noqa: E402
from ml.preprocessing.sequences import classify, hot_threshold_from_training     # noqa: E402
from ml.training.common import build_split_pool, repo_relative                   # noqa: E402

#: Sequence cap recorded in results/final/experiment_manifest.json for the committed
#: LSTM and baseline results. The cap fixes how many samples exist and therefore
#: where the chronological split falls, so it has to match for a like-for-like test set.
COMMITTED_MAX_SEQUENCES = 150_000

CLASSICAL_CSV = REPO_ROOT / "results" / "ml" / "classical_ml_comparison.csv"
LSTM_CSV = REPO_ROOT / "results" / "ml" / "transfer_learning_comparison.csv"
FULL_CSV = REPO_ROOT / "results" / "final" / "full_model_comparison.csv"

APPROACH_LABEL = {
    "linear_regression": "5. Linear regression (target only)",
    "random_forest": "6. Random forest (target only)",
    "xgboost": "7. XGBoost (target only)",
}


def score_target(trace_id: str, max_sequences: int | None, config_path: Path | None,
                 model_types: Iterable[str]) -> list[dict[str, Any]]:
    config = load_config(config_path)
    ml_config = config.ml
    seed = config.random_seed
    dataset = trace_id.split("_", 1)[0]

    # Same sequence length as the trained LSTMs, hence the same sample set and the
    # same chronological split: the comparison is only valid on identical samples.
    sequence_length = _model_sequence_length(trace_id, ml_config.sequence_length)
    pool = build_split_pool([trace_id], ml_config, max_sequences=max_sequences,
                            sequence_length=sequence_length)
    if not pool.train["targets"].size or not pool.test["targets"].size:
        print(f"  {trace_id}: empty train or test split, skipped")
        return []

    # Threshold from TRAINING targets only, the rule every other approach follows.
    threshold = hot_threshold_from_training(pool.train["targets"],
                                            ml_config.hot_percentile_cutoff)
    actual_labels = classify(pool.test["targets"], threshold)
    train_pct = round(ml_config.train_split * 100)
    val_pct = round(ml_config.val_split * 100)
    test_portion = (f"final {round(ml_config.test_split * 100)}% of the trace, "
                    f"chronologically after the {train_pct}% train and {val_pct}% validation splits")

    rows: list[dict[str, Any]] = []
    for model_type in model_types:
        print(f"  {DISPLAY_NAME[model_type]}")
        model = fit_classical(model_type, pool.train, pool.val, seed, log=print)
        predicted = model.predict_intervals(pool.test["inputs"])
        regression = regression_metrics(pool.test["targets"], predicted)
        classification = classification_metrics(
            actual_labels, classify(predicted, threshold), scores=-predicted)

        path = classical_model_path(model_type, trace_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        path.with_suffix(".json").write_text(json.dumps({
            "model_type": model_type,
            "trace_id": trace_id,
            "sequence_length": sequence_length,
            "hot_threshold_microseconds": threshold,
            "hot_threshold_source": f"target training split, p{ml_config.hot_percentile_cutoff:g}",
            "train_samples": model.train_samples,
            "training_seconds": model.training_seconds,
            "parameter_count": model.parameter_count(),
            "best_iteration": model.best_iteration,
            "seed": seed,
            "environment": environment_metadata(),
        }, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

        rows.append(_row(
            target_trace=trace_id, dataset=dataset,
            approach=APPROACH_LABEL[model_type], family="classical_ml", model_type=model_type,
            pretrained_on="none", pretrain_trace_count=0,
            finetuned_on=f"{trace_id} (training split only)",
            finetune_portion=f"first {train_pct}% of the trace", finetune_samples=model.train_samples,
            test_portion=test_portion, sequence_length=sequence_length,
            train_samples=pool.sizes["train"], val_samples=pool.sizes["val"],
            test_samples=pool.sizes["test"],
            training_seconds=model.training_seconds, parameter_count=model.parameter_count(),
            test_mae_us=regression.mae, test_rmse_us=regression.rmse,
            test_median_abs_error_us=regression.median_absolute_error,
            test_mae_log1p=regression.mae_log1p,
            hot_threshold_us=threshold,
            hot_threshold_source="training split of this model's own training data",
            **{key: getattr(classification, key) for key in
               ("accuracy", "precision", "recall", "f1", "roc_auc",
                "true_positives", "false_positives", "true_negatives",
                "false_negatives", "actual_hot_fraction", "predicted_hot_fraction")},
            checkpoint=repo_relative(path), seed=seed,
        ))
        print(f"    MAE {regression.mae:>14,.0f}  MAE(log1p) {regression.mae_log1p:.4f}  "
              f"F1 {classification.f1:.4f}")
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {repo_relative(path)}  ({len(rows)} rows)")


def merge_with_existing(classical_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Baselines + LSTMs from the existing comparison, plus the classical rows."""
    existing: list[dict[str, Any]] = []
    if LSTM_CSV.is_file():
        with LSTM_CSV.open(encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
    else:
        print(f"note: {repo_relative(LSTM_CSV)} missing; the full table has classical rows only")
    targets = {row["target_trace"] for row in classical_rows}
    kept = [row for row in existing if row["target_trace"] in targets]

    # The comparison is only valid on identical samples; refuse to merge otherwise.
    for target in targets:
        old_sizes = {int(row["test_samples"]) for row in kept if row["target_trace"] == target}
        new_sizes = {int(row["test_samples"]) for row in classical_rows
                     if row["target_trace"] == target}
        if old_sizes and old_sizes != new_sizes:
            raise RuntimeError(
                f"{target}: classical models were scored on {sorted(new_sizes)} test samples but "
                f"the existing rows use {sorted(old_sizes)}; pass --max-sequences matching "
                "results/final/experiment_manifest.json")
    return kept + classical_rows


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", default=list(CLASSICAL_MODEL_TYPES),
                        choices=list(CLASSICAL_MODEL_TYPES))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--max-sequences", type=int, default=COMMITTED_MAX_SEQUENCES,
                        help="must equal the cap the LSTM results were produced with "
                             "(results/final/experiment_manifest.json), or the test splits differ")
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows: list[dict[str, Any]] = []
    for trace_id in args.targets:
        print(f"\nScoring classical ML on {trace_id}")
        rows.extend(score_target(trace_id, args.max_sequences, args.config, args.models))
    if not rows:
        print("Nothing scored.", file=sys.stderr)
        return 1

    write_csv(rows, CLASSICAL_CSV)
    write_csv(merge_with_existing(rows), FULL_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
