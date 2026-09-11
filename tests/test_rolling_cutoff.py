"""Causality and threshold-computation checks for RollingPercentileCutoff.

These pin down the exact property the code-audit flagged as important to keep
regression-tested: classify()/stream_class() must only ever depend on
intervals observed strictly *before* the current event.
"""
import numpy as np

from ml.inference.rolling_cutoff import RollingPercentileCutoff


def test_classify_does_not_leak_the_current_event_into_its_own_threshold():
    # hot_percentile=90 (not 50): with 4 unaffected values plus one outlier in
    # a 5-deep window, the median is insensitive to a single outlier, but the
    # 90th percentile is -- needed so update() is guaranteed to move the
    # threshold, making this a meaningful before/after check.
    rc = RollingPercentileCutoff(window=5, hot_percentile=90, min_samples=1, refresh_every=1)
    for v in [1.0, 1.0, 1.0, 1.0, 1.0]:
        rc.classify(v)
        rc.update(v)

    # classify() must be a pure read: calling it must not change the
    # threshold reported by a subsequent hot_threshold() call.
    thr_before = rc.hot_threshold()
    rc.classify(999.0)
    assert rc.hot_threshold() == thr_before

    # Only update() may advance the window; after it, the threshold is
    # recomputed from the buffer *including* the just-appended outlier.
    rc.update(999.0)
    thr_after = rc.hot_threshold()
    assert thr_after != thr_before
    assert thr_after == np.percentile(np.array([1.0, 1.0, 1.0, 1.0, 999.0]), 90)


def test_warmup_uses_seed_percentile_and_then_switches_to_rolling():
    rc = RollingPercentileCutoff(window=100, hot_percentile=30, min_samples=5, refresh_every=1)
    # Below min_samples: seed percentile is applied to whatever is buffered.
    for v in [10.0, 20.0]:
        rc.update(v)
    assert rc.hot_threshold() == np.percentile(np.array([10.0, 20.0]), 30)
    for v in [30.0, 40.0, 50.0]:
        rc.update(v)
    # Now at min_samples: still a percentile over the buffer, just no longer "warm-up".
    assert rc.hot_threshold() == np.percentile(np.array([10.0, 20.0, 30.0, 40.0, 50.0]), 30)


def test_stream_edges_are_evenly_spaced_percentiles():
    rc = RollingPercentileCutoff(window=1000, hot_percentile=30, min_samples=10, refresh_every=1)
    vals = np.linspace(1, 100, 100)
    for v in vals:
        rc.update(float(v))
    edges = rc.stream_edges(3)
    assert len(edges) == 2
    assert edges[0] < edges[1]
    assert np.isclose(edges[0], np.percentile(vals, 100 / 3), rtol=0.05)
    assert np.isclose(edges[1], np.percentile(vals, 200 / 3), rtol=0.05)


def test_refresh_throttling_does_not_change_semantics_at_boundaries():
    # refresh_every=128 (production default) must give the same threshold as
    # refresh_every=1 immediately *after* a forced refresh boundary.
    vals = list(np.random.RandomState(0).uniform(1, 500, 300))
    rc_fast = RollingPercentileCutoff(window=200, hot_percentile=30, min_samples=10, refresh_every=1)
    rc_slow = RollingPercentileCutoff(window=200, hot_percentile=30, min_samples=10, refresh_every=128)
    for i, v in enumerate(vals):
        rc_fast.update(v)
        rc_slow.update(v)
        if (i + 1) % 128 == 0:
            assert rc_fast.hot_threshold() == rc_slow.hot_threshold()
