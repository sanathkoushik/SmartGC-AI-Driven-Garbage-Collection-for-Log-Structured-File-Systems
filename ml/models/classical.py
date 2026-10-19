"""Classical machine-learning predictors of the next rewrite interval.

Linear regression, random forest and XGBoost, all consuming exactly the window the
LSTM sees (the last ``L`` rewrite intervals of one LBA) and predicting the same
target (the next interval), so they can be scored on the identical test split.

Features
--------
Tree and linear models have no recurrence, so the window is summarised explicitly.
Everything is computed in ``log1p`` space, the scale the LSTM trains in, because
rewrite intervals span many orders of magnitude:

    the L raw log-intervals (oldest .. newest)
    + mean, std, min, max, median, last, last-minus-first, mean of the newest 3

The regression target is ``log1p(next interval)``; predictions are mapped back to
microseconds and clipped at zero, so metrics are directly comparable with the
LSTM's and the baselines'.

Leakage
-------
``fit`` only ever receives training (and, for early stopping, validation) data.
The hot/cold threshold is derived by the caller from training targets, exactly as
for the LSTM.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

CLASSICAL_MODEL_TYPES = ("linear_regression", "random_forest", "xgboost")

DISPLAY_NAME = {
    "linear_regression": "Linear regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
}


def build_features(inputs: np.ndarray) -> np.ndarray:
    """Window of raw intervals (N, L) -> feature matrix (N, L + 8), log1p space."""
    logged = np.log1p(np.clip(np.asarray(inputs, dtype=np.float64), 0.0, None))
    newest = logged[:, -min(3, logged.shape[1]):]
    summary = np.column_stack([
        logged.mean(axis=1),
        logged.std(axis=1),
        logged.min(axis=1),
        logged.max(axis=1),
        np.median(logged, axis=1),
        logged[:, -1],
        logged[:, -1] - logged[:, 0],
        newest.mean(axis=1),
    ])
    return np.hstack([logged, summary]).astype(np.float32)


def _make_estimator(model_type: str, seed: int) -> Any:
    if model_type == "linear_regression":
        from sklearn.linear_model import LinearRegression
        return LinearRegression()
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        # Bounded depth/leaf size keep a 100-tree forest tractable on ~10^5-10^6
        # samples without letting single leaves memorise individual LBAs.
        return RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_leaf=20,
                                     max_features=0.5, n_jobs=-1, random_state=seed)
    if model_type == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=8,
                            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                            tree_method="hist", n_jobs=-1, random_state=seed,
                            early_stopping_rounds=30, eval_metric="mae")
    raise ValueError(f"unknown classical model type {model_type!r}; "
                     f"expected one of {CLASSICAL_MODEL_TYPES}")


@dataclass
class ClassicalModel:
    """A fitted regressor plus what is needed to interpret and audit it."""

    model_type: str
    estimator: Any
    sequence_length: int
    training_seconds: float
    train_samples: int
    best_iteration: int | None = None

    def predict_intervals(self, inputs: np.ndarray) -> np.ndarray:
        """Predicted next rewrite intervals in microseconds, never negative."""
        if np.asarray(inputs).size == 0:
            return np.empty(0, dtype=np.float64)
        logged = np.asarray(self.estimator.predict(build_features(inputs)), dtype=np.float64)
        return np.clip(np.expm1(np.clip(logged, -50.0, 50.0)), 0.0, None)

    def parameter_count(self) -> int:
        """Comparable-in-spirit size: coefficients, tree nodes, or boosted-tree nodes."""
        est = self.estimator
        if self.model_type == "linear_regression":
            return int(est.coef_.size + 1)
        if self.model_type == "random_forest":
            return int(sum(tree.tree_.node_count for tree in est.estimators_))
        if self.model_type == "xgboost":
            return int(len(est.get_booster().trees_to_dataframe()))
        return 0


def fit_classical(model_type: str,
                  train: dict[str, np.ndarray],
                  val: dict[str, np.ndarray],
                  seed: int,
                  log: Callable[[str], None] | None = None) -> ClassicalModel:
    """Fit one model on the training split (validation only for early stopping)."""
    if train["targets"].size == 0:
        raise ValueError("training split is empty")

    x_train = build_features(train["inputs"])
    y_train = np.log1p(np.clip(train["targets"].astype(np.float64), 0.0, None))
    estimator = _make_estimator(model_type, seed)

    started = time.monotonic()
    best_iteration: int | None = None
    if model_type == "xgboost":
        if val["targets"].size == 0:
            raise ValueError("validation split is empty; XGBoost early stopping needs it")
        x_val = build_features(val["inputs"])
        y_val = np.log1p(np.clip(val["targets"].astype(np.float64), 0.0, None))
        estimator.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        best_iteration = int(estimator.best_iteration)
    else:
        estimator.fit(x_train, y_train)
    seconds = time.monotonic() - started

    if log:
        log(f"    fitted {DISPLAY_NAME[model_type]} on {y_train.size:,} samples in {seconds:.1f}s")
    return ClassicalModel(model_type=model_type, estimator=estimator,
                          sequence_length=int(train["inputs"].shape[1]),
                          training_seconds=seconds, train_samples=int(y_train.size),
                          best_iteration=best_iteration)
