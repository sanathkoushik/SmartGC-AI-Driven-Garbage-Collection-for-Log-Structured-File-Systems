"""Export per-write predictions for the C++ simulator.

    python -m ml.inference.predict --trace msr_prxy_0 \
        --checkpoint models/finetuned/pretrained_finetuned__msr_prxy_0.pt

Writes ``data/predictions/<model_version>__<trace_id>.csv`` with the contract
from docs/architecture.md 2.2::

    timestamp,lba,predicted_rewrite_interval,predicted_class,model_version,trace_id

Coverage and fallback
---------------------
A write can only be predicted once its LBA has accumulated ``sequence_length``
rewrite intervals, i.e. from its (L+2)-th write onward. Earlier writes get **no
row**: inventing a prediction for them would be fabricating data. The simulator
treats a write with no matching row as ``Temperature::UNKNOWN`` and places it on
the main log, and counts it in ``unpredicted_writes`` so coverage is always
visible in the results rather than assumed.

Rows are emitted in the same chronological order as the write stream they
describe, which lets the simulator merge-join them in a single pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ml.config import REPO_ROOT, load_config
from ml.evaluation.metrics import classification_metrics, regression_metrics
from ml.preprocessing.sequences import (
    build_sequences,
    classify,
    order_chronologically,
    rewrite_intervals,
)
from ml.training.common import (
    repo_relative,
    normalized_trace_path,
    load_checkpoint,
    predict_intervals,
    resolve_device,
)

PREDICTIONS_DIR = REPO_ROOT / "data" / "predictions"

PREDICTION_COLUMNS = ["timestamp", "lba", "predicted_rewrite_interval",
                      "predicted_class", "model_version", "trace_id"]


def generate_predictions(trace_id: str,
                         checkpoint_path: Path,
                         model_version: str | None = None,
                         output_dir: Path = PREDICTIONS_DIR,
                         hot_threshold: float | None = None,
                         config_path: Path | None = None,
                         device_preference: str = "auto",
                         verbose: bool = True) -> dict[str, Any]:
    """Score every predictable write of `trace_id` and write the CSV contract."""
    config = load_config(config_path)
    device = resolve_device(device_preference)
    model, scaler, metadata = load_checkpoint(checkpoint_path, device)
    sequence_length = model.hyperparameters.sequence_length
    model_version = model_version or checkpoint_path.stem

    if hot_threshold is None:
        hot_threshold = metadata.get("hot_threshold_microseconds")
    if hot_threshold is None:
        raise ValueError(
            f"no hot/cold threshold available for {checkpoint_path.name}; it must come "
            "from training data, so it cannot be derived here from the trace being scored"
        )

    path = normalized_trace_path(trace_id)
    if not path.is_file():
        raise FileNotFoundError(f"Normalized trace not found: {path}")

    frame = pd.read_csv(path, usecols=["timestamp", "lba", "operation"])
    writes = frame[frame["operation"] == "W"]
    total_writes = int(len(writes))

    intervals = rewrite_intervals(writes["lba"].to_numpy(), writes["timestamp"].to_numpy())
    samples = order_chronologically(build_sequences(intervals, sequence_length))
    n_predictable = int(samples["targets"].size)

    if n_predictable == 0:
        raise ValueError(
            f"trace '{trace_id}' yields no sequences at sequence_length={sequence_length}; "
            "there is nothing to predict"
        )

    predicted = predict_intervals(model, samples["inputs"], scaler, device)
    predicted_class = classify(predicted, hot_threshold)

    table = pd.DataFrame({
        "timestamp": samples["timestamp"].astype(np.int64),
        "lba": samples["lba"].astype(np.int64),
        "predicted_rewrite_interval": np.round(predicted, 3),
        "predicted_class": np.where(predicted_class == 1, "HOT", "COLD"),
        "model_version": model_version,
        "trace_id": trace_id,
    })[PREDICTION_COLUMNS]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_version}__{trace_id}.csv"
    table.to_csv(output_path, index=False)

    # A duplicate (timestamp, lba) would make the simulator's merge-join
    # ambiguous, so it is counted rather than assumed impossible.
    duplicate_keys = int(len(table) - table.duplicated(subset=["timestamp", "lba"]).eq(False).sum())

    # Quality of these very predictions, against the intervals that actually
    # happened. Reported over the whole trace for reference; the leakage-safe
    # held-out numbers live in the model metadata and results/ml/.
    actual = samples["targets"]
    regression = regression_metrics(actual, predicted)
    actual_class = classify(actual, hot_threshold)
    classification = classification_metrics(actual_class, predicted_class, scores=-predicted)

    summary: dict[str, Any] = {
        "trace_id": trace_id,
        "model_version": model_version,
        "checkpoint": repo_relative(checkpoint_path),
        "model_stage": metadata.get("stage"),
        "sequence_length": sequence_length,
        "hot_threshold_microseconds": float(hot_threshold),
        "hot_threshold_source": "training split of the model's own training data",
        "total_write_events": total_writes,
        "predicted_write_events": n_predictable,
        "prediction_coverage": n_predictable / total_writes if total_writes else 0.0,
        "duplicate_timestamp_lba_rows": duplicate_keys,
        "predictions_path": repo_relative(output_path),
        "whole_trace_regression_metrics_microseconds": regression.to_dict(),
        "whole_trace_classification_metrics": classification.to_dict(),
    }
    (output_dir / f"{model_version}__{trace_id}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    if verbose:
        print(f"  {model_version} on {trace_id}: {n_predictable:,} predictions "
              f"({summary['prediction_coverage']:.1%} of {total_writes:,} writes)")
        print(f"    threshold {hot_threshold:,.0f} us -> "
              f"{np.mean(predicted_class == 1):.1%} predicted HOT "
              f"(actual {classification.actual_hot_fraction:.1%})")
        print(f"    wrote {repo_relative(output_path)}")
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--output-dir", type=Path, default=PREDICTIONS_DIR)
    parser.add_argument("--hot-threshold", type=float, default=None,
                        help="override the training-derived threshold (diagnostics only)")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(list(argv) if argv is not None else None)

    generate_predictions(args.trace, args.checkpoint, model_version=args.model_version,
                         output_dir=args.output_dir, hot_threshold=args.hot_threshold,
                         config_path=args.config, device_preference=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
