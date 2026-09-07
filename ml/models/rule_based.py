"""Rung 2 - RULE_BASED: zero-training threshold heuristic.

Predicts that an LBA's next rewrite interval equals its most recently observed
inter-write gap (the ``time_since_last_write`` feature of the current event).
No parameters, no training - the honest "do you even need learning" floor above
plain MIXED placement.
"""
from __future__ import annotations

import numpy as np

from ml.models.base import BaseIntervalModel
from ml.preprocessing.features import FEATURES

_TSLW = FEATURES.index("time_since_last_write")
_ROLL = FEATURES.index("rolling_mean_interval")


class RuleBasedModel(BaseIntervalModel):
    name = "RULE_BASED"
    trainable = False

    def _raw_last_step(self, X: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.scaler["feature_mean"])
        std = np.asarray(self.scaler["feature_std"])
        return X[:, -1, :] * std + mean

    def predict_interval(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros((0,))
        raw = self._raw_last_step(X)
        recent = raw[:, _TSLW]
        rolling = raw[:, _ROLL]
        # Blend the last gap with the rolling mean so a single outlier gap does
        # not dominate; fall back to the rolling mean when the last gap is 0.
        out = np.where(recent > 0, 0.5 * recent + 0.5 * rolling, rolling)
        return np.clip(out, 1.0, None)
