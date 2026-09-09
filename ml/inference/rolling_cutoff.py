"""Phase 4b - dynamic HOT/COLD cutoff and multi-stream bucketing.

Replaces the fixed ``ml.hot_percentile_cutoff`` constant with a percentile
computed over a sliding window of the most recent predicted intervals.

Update rule (also documented in ``docs/architecture.md``)
--------------------------------------------------------
For each write event, in trace order:

1. ``classify``/``stream_class`` are evaluated against the percentile(s) of the
   intervals in the window as of the most recent *refresh* - so the cutoff only
   ever uses information available at decision time.
2. The new predicted interval is appended; the oldest is evicted once the window
   is full.
3. Until the window has at least ``min_samples`` entries the seed percentile
   from config (``hot_percentile_cutoff``) is applied to a running list instead.

For efficiency the percentiles are recomputed only every ``refresh_every``
events (and every event during warm-up); on a 400k-event real trace a per-event
``np.percentile`` over a 2000-deep window is otherwise the dominant cost. The
cutoff moves slowly, so a <= ``refresh_every`` staleness is immaterial.

HOT  = predicted interval <= P(hot_percentile_cutoff)
Stream bucket for N streams: split at the evenly spaced ``100*k/N`` percentiles
of the window.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ml.common.io_contracts import HOT, COLD, STREAM_SHORT, STREAM_LONG


class RollingPercentileCutoff:
    def __init__(self, window: int, hot_percentile: float, min_samples: int = 50,
                 refresh_every: int = 128):
        self.window = int(max(1, window))
        self.hot_p = float(hot_percentile)
        self.min_samples = int(min_samples)
        self.refresh_every = int(max(1, refresh_every))
        self._buf: deque[float] = deque(maxlen=self.window)
        self._since_refresh = self.refresh_every       # force a refresh on first use
        self._hot_thr = float("inf")
        self._edges_cache: dict[int, list[float]] = {}
        self._arr: np.ndarray | None = None

    # -- (re)compute cached thresholds -------------------------------
    def _refresh(self) -> None:
        self._since_refresh = 0
        self._edges_cache.clear()
        if not self._buf:
            self._hot_thr = float("inf")
            self._arr = None
            return
        self._arr = np.fromiter(self._buf, dtype=np.float64, count=len(self._buf))
        if len(self._buf) < self.min_samples:
            # warm-up: seed percentile on whatever we have
            self._hot_thr = float(np.percentile(self._arr, self.hot_p))
        else:
            self._hot_thr = float(np.percentile(self._arr, self.hot_p))

    def _maybe_refresh(self) -> None:
        warming = len(self._buf) < self.min_samples
        if self._since_refresh >= self.refresh_every or warming or self._arr is None:
            self._refresh()

    # -- current thresholds (also usable standalone) -----------------
    def hot_threshold(self) -> float:
        self._maybe_refresh()
        if not self._buf:
            return float("inf")
        return self._hot_thr

    def stream_edges(self, n_streams: int) -> list[float]:
        if n_streams <= 1:
            return []
        self._maybe_refresh()
        if self._arr is None or len(self._buf) < self.min_samples:
            return []
        if n_streams not in self._edges_cache:
            qs = [100.0 * k / n_streams for k in range(1, n_streams)]
            self._edges_cache[n_streams] = [float(np.percentile(self._arr, q)) for q in qs]
        return self._edges_cache[n_streams]

    # -- classify BEFORE updating (info available at decision time) --
    def classify(self, interval: float) -> int:
        return HOT if interval <= self.hot_threshold() else COLD

    def stream_class(self, interval: float, n_streams: int) -> int:
        if n_streams <= 1:
            return STREAM_SHORT
        edges = self.stream_edges(n_streams)
        if not edges:  # warm-up: fall back to binary hot/cold -> SHORT/LONG
            return STREAM_SHORT if self.classify(interval) == HOT else STREAM_LONG
        bucket = int(np.searchsorted(edges, interval, side="right"))
        return min(bucket, n_streams - 1)

    # -- then record ----------------------------------------------
    def update(self, interval: float) -> None:
        self._buf.append(float(interval))
        self._since_refresh += 1

    def state(self) -> dict:
        return {
            "window": self.window,
            "filled": len(self._buf),
            "hot_threshold": self.hot_threshold() if self._buf else None,
        }
