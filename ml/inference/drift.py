"""Phase 4b - workload-drift detector.

Tracks the distribution of predicted rewrite intervals over consecutive
non-overlapping windows of ``window`` write events.  When the symmetric KL
divergence between the previous window and the just-completed window exceeds
``kl_threshold`` the detector raises ``needs_retrain`` and records the event
index.  A moving average of |prediction error| is also tracked when ground-truth
intervals are available (offline replay), as a second, simpler drift signal.

Full online retraining is out of scope: this only detects, logs, and lets
``ml/training/retrain_now.py`` act on the flag.
"""
from __future__ import annotations

import json
import os

import numpy as np

_BINS = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, np.inf])


def _hist_p(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.ones(len(_BINS) - 1) / (len(_BINS) - 1)
    counts, _ = np.histogram(np.clip(values, 0, None), bins=_BINS)
    p = counts.astype(np.float64) + 1e-6  # Laplace smoothing
    return p / p.sum()


def sym_kl(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p))) / 2.0


class DriftDetector:
    def __init__(self, window: int, kl_threshold: float, name: str = "trace"):
        self.window = int(max(1, window))
        self.kl_threshold = float(kl_threshold)
        self.name = name
        self._cur: list[float] = []
        self._prev_p: np.ndarray | None = None
        self._err: list[float] = []
        self.timeline: list[dict] = []   # one entry per completed window
        self.needs_retrain = False
        self.last_flag_event: int | None = None
        self._event = 0

    def update(self, predicted_interval: float, true_interval: float | None = None) -> None:
        self._cur.append(float(predicted_interval))
        if true_interval is not None and np.isfinite(true_interval):
            self._err.append(abs(float(predicted_interval) - float(true_interval)))
        self._event += 1
        if len(self._cur) >= self.window:
            self._close_window()

    def _close_window(self) -> None:
        p = _hist_p(np.asarray(self._cur))
        kl = sym_kl(self._prev_p, p) if self._prev_p is not None else 0.0
        mae = float(np.mean(self._err)) if self._err else None
        flagged = self._prev_p is not None and kl > self.kl_threshold
        if flagged:
            self.needs_retrain = True
            self.last_flag_event = self._event
        self.timeline.append({
            "window_end_event": self._event,
            "kl_vs_prev": round(kl, 6),
            "pred_error_mae": None if mae is None else round(mae, 4),
            "flagged": bool(flagged),
        })
        self._prev_p = p
        self._cur.clear()
        self._err.clear()

    def finalize(self) -> None:
        if self._cur:
            self._close_window()

    def status(self) -> dict:
        return {
            "trace_name": self.name,
            "window_events": self.window,
            "kl_threshold": self.kl_threshold,
            "needs_retrain": self.needs_retrain,
            "last_flag_event": self.last_flag_event,
            "windows": len(self.timeline),
            "timeline": self.timeline,
        }

    def write_status(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.status(), fh, indent=2)
