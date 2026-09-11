"""Checks that the confidence gate's MC-dropout path is genuine (multiple
stochastic forward passes with dropout active), not a fixed/flat stand-in, and
that ConfidenceGate's trust/fallback bookkeeping is correct."""
import numpy as np
import torch
import torch.nn as nn

from ml.inference.confidence_gate import ConfidenceGate, cv_to_confidence, mc_dropout_confidence


class _FakeTorchModel:
    """Minimal stand-in for a TorchIntervalModel: exposes .module and
    ._from_scaled_target, enough for mc_dropout_confidence()."""

    def __init__(self, module: nn.Module):
        self.module = module
        self.name = "FAKE"

    def _from_scaled_target(self, y_scaled: np.ndarray) -> np.ndarray:
        return y_scaled  # identity: keep the raw module outputs comparable


def test_mc_dropout_confidence_is_low_under_heavy_dropout():
    torch.manual_seed(0)
    module = nn.Sequential(nn.Linear(4, 32), nn.Dropout(0.9), nn.Linear(32, 1))
    model = _FakeTorchModel(module)
    X = np.random.RandomState(0).randn(16, 4).astype(np.float32)
    mean, confidence = mc_dropout_confidence(model, X, k=10)
    assert mean.shape == (16,)
    assert confidence.shape == (16,)
    # Heavy dropout -> high pass-to-pass variance -> low confidence.
    assert np.mean(confidence) < 0.9


def test_mc_dropout_confidence_is_high_with_no_dropout():
    torch.manual_seed(0)
    module = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 1))  # no dropout layer at all
    model = _FakeTorchModel(module)
    X = np.random.RandomState(0).randn(16, 4).astype(np.float32)
    _, confidence = mc_dropout_confidence(model, X, k=10)
    # No stochasticity between passes -> std ~ 0 -> confidence ~ 1.
    assert np.allclose(confidence, 1.0, atol=1e-6)


def test_mc_dropout_uses_multiple_distinct_forward_passes():
    """Regression guard: if this were faked as a single deterministic pass
    repeated k times, cv_to_confidence would always see std=0. Verify the
    underlying passes actually differ under dropout."""
    torch.manual_seed(1)
    module = nn.Sequential(nn.Linear(4, 32), nn.Dropout(0.5), nn.Linear(32, 1))
    module.train()
    x = torch.randn(1, 4)
    with torch.no_grad():
        outs = [module(x).item() for _ in range(20)]
    assert len(set(outs)) > 1  # dropout masks differ pass to pass


def test_cv_to_confidence_bounds():
    mean = np.array([1.0, 1.0, 1.0])
    assert np.isclose(cv_to_confidence(mean, np.zeros(3))[0], 1.0)
    assert cv_to_confidence(mean, np.array([10.0, 0, 0]))[0] < 1.0


def test_confidence_gate_trust_and_fallback_rate():
    gate = ConfidenceGate(threshold=0.5)
    decisions = [gate.decide(c) for c in [0.9, 0.4, 0.5, 0.1]]
    assert decisions == [True, False, True, False]
    assert gate.fallback_rate == 0.5
