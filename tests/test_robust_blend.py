"""Unit tests for the HYBRID_ROBUST_SMARTGC learning-augmented blend
(ml/inference/robust_blend.py) -- the boundary behaviour is what makes it a
legitimate generalisation of ConfidenceGate's binary threshold, so it is
pinned down explicitly here."""
import numpy as np

from ml.inference.robust_blend import blend_interval, trust_weight


def test_lambda_one_always_returns_the_robust_baseline():
    model_interval = np.array([1.0, 500.0, 42.0])
    rb_interval = np.array([10.0, 10.0, 10.0])
    confidence = np.array([1.0, 1.0, 0.0])  # even full confidence must not matter
    blended, w = blend_interval(model_interval, rb_interval, confidence, lam=1.0)
    assert np.allclose(blended, rb_interval)
    assert np.allclose(w, 0.0)


def test_lambda_zero_is_pure_confidence_weighted_trust():
    model_interval = np.array([100.0])
    rb_interval = np.array([0.0])
    confidence = np.array([0.7])
    blended, w = blend_interval(model_interval, rb_interval, confidence, lam=0.0)
    assert np.isclose(w[0], 0.7)
    assert np.isclose(blended[0], 0.7 * 100.0 + 0.3 * 0.0)


def test_trust_weight_is_capped_by_one_minus_lambda_even_at_full_confidence():
    w = trust_weight(np.array([1.0]), lam=0.4)
    assert np.isclose(w[0], 0.6)  # confidence(1.0) * (1 - 0.4)


def test_trust_weight_monotonic_in_confidence_and_in_one_minus_lambda():
    conf = np.linspace(0, 1, 11)
    w_low_lambda = trust_weight(conf, lam=0.2)
    w_high_lambda = trust_weight(conf, lam=0.8)
    assert np.all(np.diff(w_low_lambda) >= 0)     # monotonic in confidence
    assert np.all(w_low_lambda >= w_high_lambda)  # more robust -> less trust, pointwise


def test_lambda_and_confidence_are_clipped_to_valid_ranges():
    blended, w = blend_interval(np.array([1.0]), np.array([0.0]), np.array([5.0]), lam=-1.0)
    assert np.isclose(w[0], 1.0)          # confidence clipped to 1, lam clipped to 0
    assert np.isclose(blended[0], 1.0)
    blended2, w2 = blend_interval(np.array([1.0]), np.array([0.0]), np.array([-5.0]), lam=2.0)
    assert np.isclose(w2[0], 0.0)         # lam clipped to 1 -> weight 0 regardless
    assert np.isclose(blended2[0], 0.0)
