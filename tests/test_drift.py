"""Sanity checks for the symmetric-KL drift detector."""
import numpy as np

from ml.inference.drift import DriftDetector, sym_kl, _hist_p


def test_sym_kl_is_zero_for_identical_distributions():
    p = _hist_p(np.array([1.0, 2.0, 4.0, 4.0, 8.0]))
    assert sym_kl(p, p) == 0.0


def test_sym_kl_is_symmetric_and_positive_for_different_distributions():
    p = _hist_p(np.full(50, 1.0))     # all short intervals
    q = _hist_p(np.full(50, 500.0))   # all long intervals
    assert sym_kl(p, q) > 0.0
    assert np.isclose(sym_kl(p, q), sym_kl(q, p))


def test_no_flag_on_a_stationary_workload():
    dd = DriftDetector(window=50, kl_threshold=0.15, name="stationary")
    rng = np.random.RandomState(0)
    for v in rng.uniform(10, 20, 500):
        dd.update(float(v))
    dd.finalize()
    assert dd.needs_retrain is False
    assert dd.last_flag_event is None


def test_flags_a_regime_shift():
    dd = DriftDetector(window=50, kl_threshold=0.15, name="shift")
    rng = np.random.RandomState(0)
    for v in rng.uniform(10, 20, 200):      # regime A
        dd.update(float(v))
    for v in rng.uniform(500, 1000, 200):   # regime B: a sharp shift
        dd.update(float(v))
    dd.finalize()
    assert dd.needs_retrain is True
    assert dd.last_flag_event is not None
    # The flag must land after the shift started (event 200), not before.
    assert dd.last_flag_event >= 200
