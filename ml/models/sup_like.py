"""Rung 3 - SUP_LIKE: the near-zero-computation SUP-GC heuristic.

SUP-GC ("Simplicity as the Ultimate Principle", ACM ToS) places every fresh
write in the hot stream and every GC-recovered valid block in the cold stream,
with no prediction at all.  The placement/migration behaviour lives entirely in
the C++ simulator (``PlacementPolicy::SUP_LIKE``); this class exists only so the
baseline ladder can emit a predictions.csv for SUP_LIKE in the common contract
(every fresh write labelled HOT / SHORT with full confidence).
"""
from __future__ import annotations

import numpy as np

from ml.models.base import BaseIntervalModel


class SupLikeModel(BaseIntervalModel):
    name = "SUP_LIKE"
    trainable = False

    def predict_interval(self, X: np.ndarray) -> np.ndarray:
        # A constant, minimal interval => the Phase 4b cutoff always labels the
        # write HOT / stream SHORT, matching "default hot on write".
        return np.ones((len(X),), dtype=np.float64)
