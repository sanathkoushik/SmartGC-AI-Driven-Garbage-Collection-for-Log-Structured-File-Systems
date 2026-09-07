"""Phase 4b - dynamic HOT/COLD cutoff and multi-stream bucketing.

Replaces the fixed ``ml.hot_percentile_cutoff`` constant with a percentile
computed over a sliding window of the most recent predicted intervals.

Update rule (also documented in ``docs/architecture.md``)
--------------------------------------------------------
For each write event, in trace order:

1. ``classify``/``stream_class`` are evaluated against the percentile(s) of the
   intervals **currently** in the window (the last ``window`` predictions,
   *before* this event is added) - so the cutoff only ever uses information
   available at decision time.
2. The new predicted interval is then appended; the oldest is evicted once the
   window is full.
3. Until the window has at least ``min_samples`` entries the seed percentile
   from config (``hot_percentile_cutoff``) is applied to a running list instead.

HOT  = predicted interval <= P(hot_percentile_cutoff)
Stream bucket for N=3 streams: SHORT / MEDIUM / LONG split at the 33rd/66th
percentiles of the window (generalised to N via evenly spaced percentiles).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ml.common.io_contracts import HOT, COLD, STREAM_SHORT, STREAM_MEDIUM, STREAM_LONG


class RollingPercentileCutoff:
    def __init__(self, window: int, hot_percentile: float, min_samples: int = 50):
        self.window = int(max(1, window))
        self.hot_p = float(hot_percentile)
        self.min_samples = int(min_samples)
        self._buf: deque[float] = deque(maxlen=self.window)

    # -- current thresholds ------------------------------------------
    def hot_threshold(self) -> float:
        if len(self._buf) < self.min_samples:
            return float("inf") if not self._buf else float(np.percentile(self._buf, self.hot_p))
        return float(np.percentile(self._buf, self.hot_p))

    def stream_edges(self, n_streams: int) -> list[float]:
        if n_streams <= 1 or len(self._buf) < self.min_samples:
            return []
        qs = [100.0 * k / n_streams for k in range(1, n_streams)]
        return [float(np.percentile(self._buf, q)) for q in qs]

    # -- classify BEFORE updating (info available at decision time) --
    def classify(self, interval: float) -> int:
        thr = self.hot_threshold()
        return HOT if interval <= thr else COLD

    def stream_class(self, interval: float, n_streams: int) -> int:
        if n_streams <= 1:
            return STREAM_SHORT
        edges = self.stream_edges(n_streams)
        if not edges:  # warmup: fall back to binary hot/cold -> SHORT/LONG
            return STREAM_SHORT if self.classify(interval) == HOT else STREAM_LONG
        bucket = int(np.searchsorted(edges, interval, side="right"))
        return min(bucket, n_streams - 1)

    # -- then record ------------------------------------------------
    def update(self, interval: float) -> None:
        self._buf.append(float(interval))

    def state(self) -> dict:
        return {
            "window": self.window,
            "filled": len(self._buf),
            "hot_threshold": self.hot_threshold() if self._buf else None,
        }
