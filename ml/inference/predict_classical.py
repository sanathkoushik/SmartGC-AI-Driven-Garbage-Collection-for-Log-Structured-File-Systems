"""Export per-write predictions of a classical ML model for the C++ simulator.

    python -m ml.inference.predict_classical --trace msr_rsrch_0 --model xgboost

Same CSV contract, coverage rule and ordering as ``ml.inference.predict`` (see its
docstring): a write gets a row only once its LBA has ``sequence_length`` prior
intervals, and rows are chronological. The model, its sequence length and the
hot/cold threshold come from ``models/classical/<model>__<trace>.joblib`` and its
JSON sidecar, which ``experiments/ml_comparison.py`` writes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

from ml.config import REPO_ROOT
from ml.inference.predict import PREDICTION_COLUMNS, PREDICTIONS_DIR
from ml.models.classical import CLASSICAL_MODEL_TYPES, ClassicalModel
from ml.preprocessing.sequences import (
    build_sequences,
    classify,
    order_chronologically,
    rewrite_intervals,
)
from ml.training.common import normalized_trace_path, repo_relative

CLASSICAL_MODELS_DIR = REPO_ROOT / "models" / "classical"


def classical_model_path(model_type: str, trace_id: str) -> Path:
    return CLASSICAL_MODELS_DIR / f"{model_type}__{trace_id}.joblib"


def generate_classical_predictions(trace_id: str,
                                   model_type: str,
                                   output_dir: Path = PREDICTIONS_DIR,
                                   verbose: bool = True) -> dict[str, Any]:
    path = classical_model_path(model_type, trace_id)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run experiments/ml_comparison.py first")
    model: ClassicalModel = joblib.load(path)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    threshold = float(metadata["hot_threshold_microseconds"])

    trace_path = normalized_trace_path(trace_id)
    if not trace_path.is_file():
        raise FileNotFoundError(f"Normalized trace not found: {trace_path}")
    frame = pd.read_csv(trace_path, usecols=["timestamp", "lba", "operation"])
    writes = frame[frame["operation"] == "W"]

    intervals = rewrite_intervals(writes["lba"].to_numpy(), writes["timestamp"].to_numpy())
    samples = order_chronologically(build_sequences(intervals, model.sequence_length))
    if samples["targets"].size == 0:
        raise ValueError(f"trace '{trace_id}' yields no sequences at "
                         f"sequence_length={model.sequence_length}")

    predicted = model.predict_intervals(samples["inputs"])
    predicted_class = classify(predicted, threshold)

    table = pd.DataFrame({
        "timestamp": samples["timestamp"].astype(np.int64),
        "lba": samples["lba"].astype(np.int64),
        "predicted_rewrite_interval": np.round(predicted, 3),
        "predicted_class": np.where(predicted_class == 1, "HOT", "COLD"),
        "model_version": model_type,
        "trace_id": trace_id,
    })[PREDICTION_COLUMNS]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_type}__{trace_id}.csv"
    table.to_csv(output_path, index=False)

    summary = {
        "trace_id": trace_id,
        "model_version": model_type,
        "model_family": "classical_ml",
        "sequence_length": model.sequence_length,
        "hot_threshold_microseconds": threshold,
        "total_write_events": int(len(writes)),
        "predicted_write_events": int(samples["targets"].size),
        "prediction_coverage": float(samples["targets"].size / max(len(writes), 1)),
        "predicted_hot_fraction": float(np.mean(predicted_class == 1)),
        "predictions_path": repo_relative(output_path),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if verbose:
        print(f"  {model_type} on {trace_id}: {summary['predicted_write_events']:,} predictions "
              f"({summary['prediction_coverage']:.1%} of {summary['total_write_events']:,} writes), "
              f"{summary['predicted_hot_fraction']:.1%} HOT")
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--model", required=True, choices=CLASSICAL_MODEL_TYPES)
    parser.add_argument("--output-dir", type=Path, default=PREDICTIONS_DIR)
    args = parser.parse_args(list(argv) if argv is not None else None)
    generate_classical_predictions(args.trace, args.model, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
