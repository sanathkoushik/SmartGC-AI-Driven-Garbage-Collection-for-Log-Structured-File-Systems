"""Rung 4 - STAT_ML: lightweight non-deep learning baseline.

A histogram gradient-boosted regressor (scikit-learn) on the same engineered
features as the LSTMs, so the matrix can isolate the *marginal* benefit of a
sequence model over cheap learning (closes gap #6).  Predicts the log1p-scaled
target, exactly like the LSTM rungs, then inverts through the shared scaler.
"""
from __future__ import annotations

import os
import pickle

import numpy as np

from ml.models.base import BaseIntervalModel


class StatMLModel(BaseIntervalModel):
    name = "STAT_ML"
    trainable = True

    def __init__(self, scaler: dict, config=None):
        super().__init__(scaler, config)
        from sklearn.ensemble import HistGradientBoostingRegressor

        mlcfg = (self.config.get("ml") or {}) if isinstance(self.config, dict) else {}
        self.model = HistGradientBoostingRegressor(
            max_iter=int(mlcfg.get("stat_ml_max_iter", 200)),
            learning_rate=float(mlcfg.get("stat_ml_learning_rate", 0.08)),
            max_depth=int(mlcfg.get("stat_ml_max_depth", 6)),
            l2_regularization=1.0,
            random_state=int(self.config.get("random_seed", 42)) if isinstance(self.config, dict) else 42,
        )

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        return X.reshape(len(X), -1) if len(X) else np.zeros((0, 1))

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(self._flatten(X_train), y_train)
        return self

    def predict_interval(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros((0,))
        y_scaled = self.model.predict(self._flatten(X))
        return self._from_scaled_target(y_scaled)

    def param_count(self) -> int:
        # Effective parameters ~= total tree node count across the ensemble.
        try:
            preds = self.model._predictors  # list[list[TreePredictor]]
            return int(sum(p.get_n_leaf_nodes() + p.get_n_leaf_nodes() - 1
                           for stage in preds for p in stage))
        except Exception:
            return int(getattr(self.model, "max_iter", 0)) * 32

    def save(self, path_dir: str) -> None:
        super().save(path_dir)
        with open(os.path.join(path_dir, "stat_ml.pkl"), "wb") as fh:
            pickle.dump(self.model, fh)

    def load(self, path_dir: str) -> "StatMLModel":
        with open(os.path.join(path_dir, "stat_ml.pkl"), "rb") as fh:
            self.model = pickle.load(fh)
        return self
