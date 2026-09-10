#!/usr/bin/env python3
"""The four-way transfer-learning comparison, on one identical test split.

    python experiments/transfer_comparison.py --targets msr_rsrch_0 msr_src2_0

For every target workload this scores, on the **same held-out chronological test
split**:

1. the non-learned baselines (last interval, mean, median, moving average, EMA)
2. an LSTM trained from scratch on that target alone
3. the MSR-pretrained LSTM applied **without any adaptation**
4. the MSR-pretrained LSTM **after fine-tuning** on the target's early data

and records, for each, the provenance that makes the number interpretable: which
workloads were pretrained on, which workload and which chronological portion were
used for adaptation, how many samples and how long training took, and which
portion of the target was held back for the test.

The test split is read exactly once, here, at scoring time. It is never seen
during training, checkpoint selection or threshold derivation - see
docs/methodology.md.

Writes ``results/ml/transfer_learning_comparison.csv``.
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
    build_split_pool,
    load_checkpoint,
    predict_intervals,
    resolve_device,
)
from ml.training.finetune import MODEL_TYPES, default_output_path       # noqa: E402

OUTPUT_PATH = REPO_ROOT / "results" / "ml" / "transfer_learning_comparison.csv"

COLUMNS = [
    # identity
    "target_trace", "dataset", "approach", "family", "model_type",
    # provenance
    "pretrained_on", "pretrain_trace_count", "finetuned_on",
    "finetune_portion", "finetune_samples", "test_portion",
    # data sizes
    "sequence_length", "train_samples", "val_samples", "test_samples",
    # cost
    "training_seconds", "parameter_count",
    # regression on the held-out test split
    "test_mae_us", "test_rmse_us", "test_median_abs_error_us", "test_mae_log1p",
    # hot/cold classification on the same split
    "hot_threshold_us", "hot_threshold_source",
    "accuracy", "precision", "recall", "f1", "roc_auc",
    "true_positives", "false_positives", "true_negatives", "false_negatives",
    "actual_hot_fraction", "predicted_hot_fraction",
    # bookkeeping
    "checkpoint", "seed",
]

APPROACH_LABEL = {
    "scratch": "2. LSTM from scratch (target only)",
    "pretrained": "3. Pretrained MSR LSTM, no fine-tuning",
    "pretrained_finetuned": "4. Pretrained MSR LSTM + target fine-tuning",
}


def _row(**kwargs: Any) -> dict[str, Any]:
    row = {column: "" for column in COLUMNS}
    row.update(kwargs)
    return row


def _model_sequence_length(trace_id: str, fallback: int) -> int:
    """Sequence length the trained models for `trace_id` use.

    Read from a checkpoint rather than from config.yaml, because the length is
    chosen by the hyperparameter search and the config default may differ.
    """
    for model_type in MODEL_TYPES:
        checkpoint = default_output_path(model_type, trace_id)
        if checkpoint.is_file():
            model, _, _ = load_checkpoint(checkpoint)
            return int(model.hyperparameters.sequence_length)
    return fallback


def compare_target(trace_id: str,
                   config_path: Path | None = None,
                   max_sequences: int | None = DEFAULT_MAX_SEQUENCES_PER_TRACE,
                   device_preference: str = "auto",
                   verbose: bool = True) -> list[dict[str, Any]]:
    """Score all four approaches for one target on its single held-out test split."""
    config = load_config(config_path)
    ml_config = config.ml
    device = resolve_device(device_preference)
    dataset = trace_id.split("_", 1)[0]
    rows: list[dict[str, Any]] = []

    train_pct = round(ml_config.train_split * 100)
    val_pct = round(ml_config.val_split * 100)
    test_portion = (f"final {round(ml_config.test_split * 100)}% of the trace, "
                    f"chronologically after the {train_pct}% train and {val_pct}% validation splits")

    pools: dict[int, Any] = {}

    def pool_for(sequence_length: int):
        if sequence_length not in pools:
            pools[sequence_length] = build_split_pool(
                [trace_id], ml_config, max_sequences=max_sequences,
                sequence_length=sequence_length)
        return pools[sequence_length]

    # ---- 1. non-learned baselines ----------------------------------------
    # Scored at the *models'* sequence length, so every approach in this table
    # sees the identical test split. Building the baseline pool at the config
    # default instead would silently compare them on a different set of samples:
    # a different window length yields a different number of sequences and so a
    # different chronological split.
    model_sequence_length = _model_sequence_length(trace_id, ml_config.sequence_length)
    base_pool = pool_for(model_sequence_length)
    if base_pool.test["targets"].size and base_pool.train["targets"].size:
        # Same rule the models follow: the cutoff comes from TRAINING targets.
        threshold = hot_threshold_from_training(base_pool.train["targets"],
                                                ml_config.hot_percentile_cutoff)
        actual_labels = classify(base_pool.test["targets"], threshold)

        for name, predictor in BASELINE_PREDICTORS.items():
            predicted = predictor(base_pool.test["inputs"])
            regression = regression_metrics(base_pool.test["targets"], predicted)
            classification = classification_metrics(
                actual_labels, classify(predicted, threshold), scores=-predicted)
            rows.append(_row(
                target_trace=trace_id, dataset=dataset,
                approach=f"1. Baseline: {name}", family="baseline", model_type=name,
                pretrained_on="none", pretrain_trace_count=0,
                finetuned_on="none (no training of any kind)",
                finetune_portion="n/a", finetune_samples=0,
                test_portion=test_portion,
                sequence_length=model_sequence_length,
                train_samples=base_pool.sizes["train"], val_samples=base_pool.sizes["val"],
                test_samples=base_pool.sizes["test"],
                training_seconds=0.0, parameter_count=0,
                test_mae_us=regression.mae, test_rmse_us=regression.rmse,
                test_median_abs_error_us=regression.median_absolute_error,
                test_mae_log1p=regression.mae_log1p,
                hot_threshold_us=threshold,
                hot_threshold_source=f"target training split, p{ml_config.hot_percentile_cutoff:g}",
                **{key: getattr(classification, key) for key in
                   ("accuracy", "precision", "recall", "f1", "roc_auc",
                    "true_positives", "false_positives", "true_negatives",
                    "false_negatives", "actual_hot_fraction", "predicted_hot_fraction")},
                checkpoint="", seed=config.random_seed,
            ))

    # ---- 2/3/4. the three model variants ---------------------------------
    for model_type in MODEL_TYPES:
        checkpoint = default_output_path(model_type, trace_id)
        if not checkpoint.is_file():
            if verbose:
                print(f"    {model_type}: no checkpoint yet, skipping")
            continue

        model, scaler, metadata = load_checkpoint(checkpoint, device)
        sequence_length = model.hyperparameters.sequence_length
        pool = pool_for(sequence_length)
        if pool.test["targets"].size == 0:
            continue

        threshold = float(metadata.get("hot_threshold_microseconds", 0.0))
        if threshold <= 0:
            raise ValueError(f"{checkpoint.name} records no training-derived hot threshold")

        predicted = predict_intervals(model, pool.test["inputs"], scaler, device)
        regression = regression_metrics(pool.test["targets"], predicted)
        actual_labels = classify(pool.test["targets"], threshold)
        classification = classification_metrics(
            actual_labels, classify(predicted, threshold), scores=-predicted)

        pretrain_traces = metadata.get("pretrain_traces") or []
        if model_type == "scratch":
            pretrained_on, finetuned_on = "none", f"{trace_id} (training split only)"
            finetune_portion = f"first {train_pct}% of the trace"
        elif model_type == "pretrained":
            pretrained_on = ", ".join(pretrain_traces) or "unknown"
            finetuned_on = "none - applied unchanged"
            finetune_portion = "n/a"
        else:
            pretrained_on = ", ".join(pretrain_traces) or "unknown"
            finetuned_on = f"{trace_id} (training split only)"
            fraction = metadata.get("finetune_fraction_of_train_split", 1.0)
            finetune_portion = (f"first {train_pct}% of the trace"
                                + (f", of which the earliest {fraction:.0%}" if fraction < 1.0 else ""))

        rows.append(_row(
            target_trace=trace_id, dataset=dataset,
            approach=APPROACH_LABEL[model_type], family="lstm", model_type=model_type,
            pretrained_on=pretrained_on, pretrain_trace_count=len(pretrain_traces),
            finetuned_on=finetuned_on, finetune_portion=finetune_portion,
            finetune_samples=metadata.get("samples", {}).get("used_for_adaptation", 0),
            test_portion=test_portion,
            sequence_length=sequence_length,
            train_samples=pool.sizes["train"], val_samples=pool.sizes["val"],
            test_samples=pool.sizes["test"],
            training_seconds=metadata.get("training_seconds", 0.0),
            parameter_count=metadata.get("parameter_count", 0),
            test_mae_us=regression.mae, test_rmse_us=regression.rmse,
            test_median_abs_error_us=regression.median_absolute_error,
            test_mae_log1p=regression.mae_log1p,
            hot_threshold_us=threshold,
            hot_threshold_source="training split of this model's own training data",
            **{key: getattr(classification, key) for key in
               ("accuracy", "precision", "recall", "f1", "roc_auc",
                "true_positives", "false_positives", "true_negatives",
                "false_negatives", "actual_hot_fraction", "predicted_hot_fraction")},
            checkpoint=str(checkpoint.relative_to(REPO_ROOT)).replace("\\", "/"),
            seed=metadata.get("seed", config.random_seed),
        ))

        if verbose:
            print(f"    {APPROACH_LABEL[model_type]:<44} "
                  f"MAE {regression.mae:>14,.0f}  MAE(log1p) {regression.mae_log1p:.4f}  "
                  f"F1 {classification.f1:.4f}")

    return rows


def write_comparison(rows: Sequence[dict[str, Any]], path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {path.relative_to(REPO_ROOT)}  ({len(rows)} rows)")


def print_summary(rows: Sequence[dict[str, Any]]) -> None:
    """One block per target: all four approaches side by side, measured values only."""
    targets = sorted({row["target_trace"] for row in rows})
    for trace_id in targets:
        subset = [row for row in rows if row["target_trace"] == trace_id]
        lstm = [row for row in subset if row["family"] == "lstm"]
        if len(lstm) < len(MODEL_TYPES):
            print(f"\n{trace_id}: only {len(lstm)} of {len(MODEL_TYPES)} model variants "
                  "present - no conclusion drawn.")
            continue

        print(f"\n{'=' * 96}\n  {trace_id}   (test split: {subset[0]['test_portion']})\n{'=' * 96}")
        print(f"  {'Approach':<46} {'MAE (us)':>15} {'RMSE (us)':>16} {'MAE log1p':>10} {'F1':>8}")
        print("  " + "-" * 94)
        for row in sorted(subset, key=lambda r: r["approach"]):
            print(f"  {row['approach']:<46} {row['test_mae_us']:>15,.0f} "
                  f"{row['test_rmse_us']:>16,.0f} {row['test_mae_log1p']:>10.4f} "
                  f"{row['f1']:>8.4f}")

        best_baseline = min((row for row in subset if row["family"] == "baseline"),
                            key=lambda r: r["test_mae_log1p"], default=None)
        by_type = {row["model_type"]: row for row in lstm}
        scratch = by_type["scratch"]
        pretrained = by_type["pretrained"]
        finetuned = by_type["pretrained_finetuned"]

        print("\n  Measured differences on this split (log1p MAE, lower is better):")
        if best_baseline:
            print(f"    best baseline ({best_baseline['model_type']}): "
                  f"{best_baseline['test_mae_log1p']:.4f}")
        print(f"    scratch                                : {scratch['test_mae_log1p']:.4f}")
        print(f"    pretrained (no fine-tuning)            : {pretrained['test_mae_log1p']:.4f}"
              f"   ({pretrained['test_mae_log1p'] - scratch['test_mae_log1p']:+.4f} vs scratch)")
        print(f"    pretrained + fine-tuned                : {finetuned['test_mae_log1p']:.4f}"
              f"   ({finetuned['test_mae_log1p'] - pretrained['test_mae_log1p']:+.4f} vs pretrained)")
        print(f"\n  Training cost: scratch {scratch['training_seconds'] / 60:.1f} min, "
              f"fine-tuning {finetuned['training_seconds'] / 60:.1f} min "
              f"(pretraining is shared and not recharged per target)")
        print(f"  Samples: {scratch['train_samples']:,} train / "
              f"{scratch['val_samples']:,} validation / {scratch['test_samples']:,} test")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", nargs="*", default=None,
                        help="defaults to the targets and external workloads in dataset_roles.json")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--max-sequences", type=int, default=DEFAULT_MAX_SEQUENCES_PER_TRACE)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(list(argv) if argv is not None else None)

    targets = args.targets
    if not targets:
        roles_path = REPO_ROOT / "results" / "dataset_roles.json"
        if not roles_path.is_file():
            print("results/dataset_roles.json missing; pass --targets explicitly",
                  file=sys.stderr)
            return 2
        roles = json.loads(roles_path.read_text(encoding="utf-8"))
        targets = list(roles.get("targets", [])) + list(roles.get("external", []))

    rows: list[dict[str, Any]] = []
    for trace_id in targets:
        print(f"\nScoring {trace_id} on its held-out test split")
        rows.extend(compare_target(trace_id, config_path=args.config,
                                   max_sequences=args.max_sequences,
                                   device_preference=args.device))

    if not rows:
        print("Nothing to compare.", file=sys.stderr)
        return 1
    write_comparison(rows, args.output)
    print_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
