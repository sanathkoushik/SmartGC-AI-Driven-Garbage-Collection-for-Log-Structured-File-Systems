"""Model shape, determinism, checkpoint round-trip, training, and baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ml.config import MlConfig
from ml.evaluation.baselines import BASELINE_PREDICTORS, exponential_moving_average, moving_average
from ml.evaluation.metrics import classification_metrics, regression_metrics
from ml.models.lstm import LstmHyperparameters, build_model, make_loss
from ml.preprocessing.sequences import IntervalScaler
from ml.training.common import (
    SplitPool,
    load_checkpoint,
    predict_intervals,
    save_checkpoint,
    set_deterministic_seed,
    train_model,
)


def _pool(n_train: int = 512, n_val: int = 128, n_test: int = 128, length: int = 6) -> SplitPool:
    """A learnable toy task: the next interval is close to the window mean."""
    rng = np.random.default_rng(11)

    def make(n: int, start: int) -> dict[str, np.ndarray]:
        base = rng.uniform(10.0, 10_000.0, size=n)
        inputs = np.clip(base[:, None] * rng.uniform(0.8, 1.2, size=(n, length)), 1.0, None)
        targets = inputs.mean(axis=1)
        return {"inputs": inputs.astype(np.float32),
                "targets": targets.astype(np.float32),
                "timestamp": np.arange(start, start + n, dtype=np.int64),
                "lba": rng.integers(0, 50, size=n)}

    return SplitPool(train=make(n_train, 0), val=make(n_val, n_train),
                     test=make(n_test, n_train + n_val))


def test_model_forward_shape():
    model = build_model(LstmHyperparameters(sequence_length=7, hidden_dim=16, num_layers=2))
    out2d = model(torch.zeros(5, 7))
    out3d = model(torch.zeros(5, 7, 1))
    assert out2d.shape == (5,)
    assert out3d.shape == (5,)
    assert model.parameter_count() > 0


def test_single_layer_model_has_no_inter_layer_dropout():
    # PyTorch warns and ignores dropout with one layer; the model must not set it.
    model = build_model(LstmHyperparameters(num_layers=1, dropout=0.5))
    assert model.lstm.dropout == 0.0


def test_model_initialisation_is_seed_deterministic():
    hyperparameters = LstmHyperparameters(sequence_length=4, hidden_dim=8, num_layers=1)
    a = build_model(hyperparameters, seed=123)
    b = build_model(hyperparameters, seed=123)
    c = build_model(hyperparameters, seed=124)

    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
    assert any(not torch.equal(pa, pc) for pa, pc in zip(a.parameters(), c.parameters()))


def test_make_loss_rejects_unknown_name():
    assert make_loss("SmoothL1") is not None
    with pytest.raises(ValueError):
        make_loss("Huberish")


def test_training_updates_weights_and_improves_validation():
    pool = _pool()
    config = MlConfig(sequence_length=6, hidden_dim=16, num_layers=1, batch_size=64,
                      epochs=6, learning_rate=5e-3, early_stopping_patience=6)
    scaler = IntervalScaler.fit(pool.train["inputs"], pool.train["targets"])
    model = build_model(LstmHyperparameters(sequence_length=6, hidden_dim=16, num_layers=1),
                        seed=0)
    before = [p.detach().clone() for p in model.parameters()]

    result = train_model(model, pool, scaler, config, seed=0, device=torch.device("cpu"))

    assert result.epochs_run >= 1
    assert 1 <= result.best_epoch <= result.epochs_run
    assert any(not torch.equal(b, a.detach()) for b, a in zip(before, model.parameters()))
    # Loss must fall from the first epoch to the best one on a learnable task.
    assert result.best_val_loss < result.history[0]["val_loss"]


def test_training_is_reproducible_with_a_fixed_seed():
    pool = _pool(n_train=256, n_val=64, n_test=64)
    config = MlConfig(sequence_length=6, hidden_dim=8, num_layers=1, batch_size=32,
                      epochs=3, learning_rate=5e-3, early_stopping_patience=3)
    scaler = IntervalScaler.fit(pool.train["inputs"], pool.train["targets"])

    def run() -> list[float]:
        set_deterministic_seed(5)
        model = build_model(LstmHyperparameters(sequence_length=6, hidden_dim=8, num_layers=1),
                            seed=5)
        result = train_model(model, pool, scaler, config, seed=5, device=torch.device("cpu"))
        return [entry["val_loss"] for entry in result.history]

    assert run() == run()


def test_training_requires_a_validation_split():
    pool = _pool()
    empty = {k: v[:0] for k, v in pool.val.items()}
    config = MlConfig(epochs=1)
    scaler = IntervalScaler.fit(pool.train["inputs"], pool.train["targets"])
    model = build_model(LstmHyperparameters(sequence_length=6, hidden_dim=8, num_layers=1))

    with pytest.raises(ValueError):
        train_model(model, SplitPool(train=pool.train, val=empty, test=pool.test),
                    scaler, config, seed=0, device=torch.device("cpu"))


def test_checkpoint_round_trip_preserves_predictions(tmp_path: Path):
    pool = _pool(n_train=128, n_val=32, n_test=32)
    scaler = IntervalScaler.fit(pool.train["inputs"], pool.train["targets"])
    hyperparameters = LstmHyperparameters(sequence_length=6, hidden_dim=12, num_layers=2,
                                          dropout=0.1)
    model = build_model(hyperparameters, seed=3)
    device = torch.device("cpu")

    before = predict_intervals(model, pool.test["inputs"], scaler, device)
    path = tmp_path / "model.pt"
    save_checkpoint(model, scaler, {"stage": "unit-test",
                                    "hot_threshold_microseconds": 123.0}, path)

    restored, restored_scaler, metadata = load_checkpoint(path, device)
    after = predict_intervals(restored, pool.test["inputs"], restored_scaler, device)

    assert np.allclose(before, after)
    assert restored.hyperparameters == hyperparameters
    assert restored_scaler.mean == scaler.mean and restored_scaler.std == scaler.std
    assert metadata["stage"] == "unit-test"
    assert metadata["parameter_count"] == model.parameter_count()
    assert "environment" in metadata


def test_predictions_are_non_negative_microseconds():
    pool = _pool(n_train=64, n_val=16, n_test=16)
    scaler = IntervalScaler.fit(pool.train["inputs"], pool.train["targets"])
    model = build_model(LstmHyperparameters(sequence_length=6, hidden_dim=8, num_layers=1),
                        seed=1)
    predictions = predict_intervals(model, pool.test["inputs"], scaler, torch.device("cpu"))
    assert predictions.shape == (16,)
    assert (predictions >= 0).all()
    assert np.isfinite(predictions).all()


def test_predict_intervals_handles_empty_input():
    scaler = IntervalScaler(mean=0.0, std=1.0, fitted=True)
    model = build_model(LstmHyperparameters(sequence_length=4, hidden_dim=8, num_layers=1))
    assert predict_intervals(model, np.empty((0, 4)), scaler, torch.device("cpu")).size == 0


# ---------------------------------------------------------------------------
# Baselines and metrics
# ---------------------------------------------------------------------------

def test_baselines_produce_expected_values():
    window = np.array([[1.0, 2.0, 3.0, 10.0]])
    assert BASELINE_PREDICTORS["last_interval"](window)[0] == 10.0
    assert BASELINE_PREDICTORS["mean_interval"](window)[0] == 4.0
    assert BASELINE_PREDICTORS["median_interval"](window)[0] == 2.5
    assert BASELINE_PREDICTORS["moving_average_3"](window)[0] == pytest.approx(5.0)

    # EMA weights sum to one, so a constant window returns that constant.
    constant = np.full((1, 5), 7.0)
    assert exponential_moving_average(0.5)(constant)[0] == pytest.approx(7.0)
    # A shorter window than requested is handled without error.
    assert moving_average(10)(window)[0] == pytest.approx(4.0)


def test_baseline_registry_shapes():
    inputs = np.random.default_rng(0).uniform(1, 100, size=(20, 8))
    for name, predictor in BASELINE_PREDICTORS.items():
        out = predictor(inputs)
        assert out.shape == (20,), name
        assert np.isfinite(out).all(), name


def test_regression_metrics_are_correct():
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([12.0, 18.0, 33.0])
    metrics = regression_metrics(actual, predicted)
    assert metrics.mae == pytest.approx((2 + 2 + 3) / 3)
    assert metrics.rmse == pytest.approx(np.sqrt((4 + 4 + 9) / 3))
    assert metrics.median_absolute_error == pytest.approx(2.0)
    assert metrics.n == 3


def test_classification_metrics_are_correct():
    actual = np.array([1, 1, 0, 0, 1])
    predicted = np.array([1, 0, 0, 1, 1])
    metrics = classification_metrics(actual, predicted, scores=np.array([0.9, 0.2, 0.1, 0.8, 0.7]))

    assert metrics.true_positives == 2
    assert metrics.false_negatives == 1
    assert metrics.false_positives == 1
    assert metrics.true_negatives == 1
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.f1 == pytest.approx(2 / 3)
    assert metrics.accuracy == pytest.approx(3 / 5)
    assert metrics.roc_auc is not None


def test_classification_auc_is_none_for_a_single_class():
    metrics = classification_metrics(np.array([1, 1, 1]), np.array([1, 0, 1]),
                                     scores=np.array([0.1, 0.2, 0.3]))
    assert metrics.roc_auc is None
    assert metrics.actual_hot_fraction == 1.0
