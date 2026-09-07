"""Common interface for every rung of the Phase 4a baseline ladder.

All models expose the same contract: given a sequence tensor ``X`` of shape
``[N, seq_len, F]`` (already scaled by ``ml/models/scaler.json``) they return a
**raw** predicted next-rewrite-interval per sample.  The Phase 4b inference
layer is solely responsible for turning intervals into HOT/COLD labels and
stream-class buckets, so the whole ladder is compared under one protocol.
"""
from __future__ import annotations

import abc
import json
import os
import time
from typing import Optional

import numpy as np

from ml.preprocessing.features import inverse_target, transform_target


class BaseIntervalModel(abc.ABC):
    name: str = "base"
    trainable: bool = True

    def __init__(self, scaler: dict, config: Optional[dict] = None):
        self.scaler = scaler
        self.config = config or {}

    # -- lifecycle ---------------------------------------------------------
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> "BaseIntervalModel":
        return self

    @abc.abstractmethod
    def predict_interval(self, X: np.ndarray) -> np.ndarray:
        """Raw (un-transformed) next-rewrite-interval prediction, shape [N]."""

    # -- cost accounting (Phase 7) --------------------------------------
    def param_count(self) -> int:
        return 0

    def measure_latency(self, X: np.ndarray, batch_size: int = 256, repeats: int = 3) -> dict:
        if len(X) == 0:
            return {"batches": 0, "latency_ms_per_batch": 0.0, "throughput_pred_per_s": 0.0}
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            for i in range(0, len(X), batch_size):
                self.predict_interval(X[i:i + batch_size])
            best = min(best, time.perf_counter() - t0)
        n_batches = int(np.ceil(len(X) / batch_size))
        return {
            "batches": n_batches,
            "latency_ms_per_batch": 1000.0 * best / n_batches,
            "throughput_pred_per_s": len(X) / best if best > 0 else 0.0,
        }

    # -- persistence -----------------------------------------------------
    def save(self, path_dir: str) -> None:
        os.makedirs(path_dir, exist_ok=True)
        with open(os.path.join(path_dir, "model_meta.json"), "w", encoding="utf-8") as fh:
            json.dump({"name": self.name, "param_count": self.param_count()}, fh, indent=2)

    def load(self, path_dir: str) -> "BaseIntervalModel":
        return self

    # -- helpers for torch subclasses ---------------------------------
    def _to_scaled_target(self, y_raw: np.ndarray) -> np.ndarray:
        return transform_target(y_raw, self.scaler)

    def _from_scaled_target(self, y_scaled: np.ndarray) -> np.ndarray:
        return np.clip(inverse_target(y_scaled, self.scaler), 0.0, None)
