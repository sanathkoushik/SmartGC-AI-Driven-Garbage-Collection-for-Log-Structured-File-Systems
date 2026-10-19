"""Classical ML predictors: features, fitting, leakage-safety and the export contract."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from ml.evaluation.baselines import median_interval                       # noqa: E402
from ml.evaluation.metrics import regression_metrics                      # noqa: E402
from ml.inference import predict_classical                                # noqa: E402
from ml.models.classical import (                                         # noqa: E402
    CLASSICAL_MODEL_TYPES,
    build_features,
    fit_classical,
)

LENGTH = 6


def _split(n: int, start: int, seed: int) -> dict[str, np.ndarray]:
    """Learnable toy task a median/mean heuristic cannot solve.

    Windows vary widely across their positions, and the next interval is three
    times the *oldest* one, so the answer is in the window but not in its centre.
    """
    rng = np.random.default_rng(seed)
    inputs = np.exp(rng.uniform(np.log(10.0), np.log(100_000.0), size=(n, LENGTH)))
    return {"inputs": inputs.astype(np.float32),
            "targets": (3.0 * inputs[:, 0]).astype(np.float32),
            "timestamp": np.arange(start, start + n, dtype=np.int64),
            "lba": rng.integers(0, 50, size=n)}


@pytest.fixture(scope="module")
def pools() -> tuple[dict, dict, dict]:
    return _split(1500, 0, 1), _split(400, 1500, 2), _split(400, 1900, 3)


@pytest.fixture(scope="module")
def fitted(pools) -> dict:
    train, val, _ = pools
    return {name: fit_classical(name, train, val, seed=7) for name in CLASSICAL_MODEL_TYPES}


def test_features_shape_and_finiteness():
    window = np.random.default_rng(0).uniform(0, 1e6, size=(32, LENGTH))
    features = build_features(window)
    assert features.shape == (32, LENGTH + 8)
    assert np.isfinite(features).all()


def test_features_are_per_row_and_do_not_mutate_input():
    window = np.random.default_rng(0).uniform(0, 1e6, size=(32, LENGTH))
    before = window.copy()
    features = build_features(window)
    assert np.array_equal(window, before)
    # No cross-row statistics: reordering rows reorders features identically.
    order = np.random.default_rng(1).permutation(32)
    assert np.allclose(build_features(window[order]), features[order])


def test_zero_intervals_are_valid_input():
    """Two writes in the same microsecond give a genuine zero interval."""
    features = build_features(np.zeros((4, LENGTH)))
    assert np.isfinite(features).all()


@pytest.mark.parametrize("model_type", CLASSICAL_MODEL_TYPES)
def test_predictions_are_finite_nonnegative_microseconds(fitted, pools, model_type):
    predicted = fitted[model_type].predict_intervals(pools[2]["inputs"])
    assert predicted.shape == pools[2]["targets"].shape
    assert np.isfinite(predicted).all() and (predicted >= 0).all()


@pytest.mark.parametrize("model_type", CLASSICAL_MODEL_TYPES)
def test_models_beat_the_median_baseline_on_a_learnable_task(fitted, pools, model_type):
    test = pools[2]
    model_error = regression_metrics(
        test["targets"], fitted[model_type].predict_intervals(test["inputs"])).mae_log1p
    baseline_error = regression_metrics(
        test["targets"], median_interval(test["inputs"])).mae_log1p
    assert model_error < baseline_error


def test_empty_input_returns_empty(fitted):
    out = fitted["linear_regression"].predict_intervals(np.empty((0, LENGTH), dtype=np.float32))
    assert out.size == 0


def test_unknown_model_type_is_rejected(pools):
    with pytest.raises(ValueError, match="unknown classical model type"):
        fit_classical("svm", pools[0], pools[1], seed=1)


def test_empty_training_split_is_rejected(pools):
    empty = {key: value[:0] for key, value in pools[0].items()}
    with pytest.raises(ValueError, match="training split is empty"):
        fit_classical("linear_regression", empty, pools[1], seed=1)


def test_xgboost_requires_a_validation_split(pools):
    empty = {key: value[:0] for key, value in pools[1].items()}
    with pytest.raises(ValueError, match="validation split is empty"):
        fit_classical("xgboost", pools[0], empty, seed=1)


@pytest.mark.parametrize("model_type", ["linear_regression", "random_forest"])
def test_validation_data_cannot_influence_models_that_do_not_use_it(pools, model_type):
    """Leakage guard: only XGBoost may look at validation (for early stopping)."""
    train, val, test = pools
    shifted_val = dict(val, targets=(val["targets"] * 50.0).astype(np.float32))
    a = fit_classical(model_type, train, val, seed=3).predict_intervals(test["inputs"])
    b = fit_classical(model_type, train, shifted_val, seed=3).predict_intervals(test["inputs"])
    assert np.allclose(a, b, rtol=1e-9, atol=0.0)


def test_fitting_is_deterministic_for_a_seed(pools):
    """Same seed, same forest. Not bit-identical: averaging trees across threads
    changes the summation order, which moves results by ~1e-14 relative."""
    train, val, test = pools
    a = fit_classical("random_forest", train, val, seed=5).predict_intervals(test["inputs"])
    b = fit_classical("random_forest", train, val, seed=5).predict_intervals(test["inputs"])
    assert np.allclose(a, b, rtol=1e-9, atol=0.0)


def test_parameter_counts_are_positive(fitted):
    assert all(model.parameter_count() > 0 for model in fitted.values())
    assert fitted["linear_regression"].parameter_count() == LENGTH + 8 + 1


# ---------------------------------------------------------------------------
# Prediction export (the CSV contract the C++ simulator reads)
# ---------------------------------------------------------------------------

@pytest.fixture()
def exported(tmp_path, monkeypatch, fitted):
    """Write a tiny normalized trace and export predictions for it."""
    rng = np.random.default_rng(4)
    n = 400
    trace = pd.DataFrame({
        "timestamp": np.sort(rng.integers(0, 10_000_000, size=n)).astype(np.int64),
        "lba": rng.integers(0, 8, size=n),
        "size": 4096,
        "operation": "W",
        "trace_id": "toy",
    })
    trace_path = tmp_path / "toy.csv"
    trace.to_csv(trace_path, index=False)

    model_path = tmp_path / "linear_regression__toy.joblib"
    joblib.dump(fitted["linear_regression"], model_path)
    model_path.with_suffix(".json").write_text(
        json.dumps({"hot_threshold_microseconds": 1_000_000.0}), encoding="utf-8")

    monkeypatch.setattr(predict_classical, "classical_model_path", lambda m, t: model_path)
    monkeypatch.setattr(predict_classical, "normalized_trace_path", lambda t: trace_path)
    summary = predict_classical.generate_classical_predictions(
        "toy", "linear_regression", output_dir=tmp_path, verbose=False)
    return trace, summary, tmp_path / "linear_regression__toy.csv"


def test_export_has_the_documented_columns(exported):
    _, _, path = exported
    with path.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    assert header == ["timestamp", "lba", "predicted_rewrite_interval",
                      "predicted_class", "model_version", "trace_id"]


def test_export_is_chronological_and_labelled(exported):
    _, _, path = exported
    table = pd.read_csv(path)
    assert (np.diff(table["timestamp"].to_numpy()) >= 0).all()
    assert set(table["predicted_class"]) <= {"HOT", "COLD"}
    assert (table["predicted_rewrite_interval"] >= 0).all()
    assert set(table["model_version"]) == {"linear_regression"}


def test_export_only_covers_writes_with_enough_history(exported):
    """A write is predictable only once its LBA has LENGTH prior intervals."""
    trace, summary, path = exported
    table = pd.read_csv(path)
    expected = int((trace.groupby("lba").size() - (LENGTH + 1)).clip(lower=0).sum())
    assert len(table) == summary["predicted_write_events"] == expected
    assert summary["prediction_coverage"] < 1.0


# ---------------------------------------------------------------------------
# Comparison guard: refuse to merge results scored on different test sets
# ---------------------------------------------------------------------------

def test_merge_refuses_mismatched_test_sets(tmp_path, monkeypatch):
    from experiments import ml_comparison

    existing = tmp_path / "existing.csv"
    with existing.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["target_trace", "test_samples"])
        writer.writeheader()
        writer.writerow({"target_trace": "t", "test_samples": 100})
    monkeypatch.setattr(ml_comparison, "LSTM_CSV", existing)

    with pytest.raises(RuntimeError, match="test samples"):
        ml_comparison.merge_with_existing([{"target_trace": "t", "test_samples": 250}])
    merged = ml_comparison.merge_with_existing([{"target_trace": "t", "test_samples": 100}])
    assert len(merged) == 2
