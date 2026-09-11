"""Shape/behaviour checks for the LSTM+attention module, including the
``ml.attention.enabled`` flag fix (previously a dead config key -- the module
always built attention regardless of its value)."""
import torch

from ml.models.lstm_attention import _make_attn_module


def test_forward_shape_with_attention_enabled():
    m = _make_attn_module(n_features=5, hidden_dim=8, num_layers=2, dropout=0.1,
                           num_heads=2, attention_enabled=True)
    x = torch.randn(3, 10, 5)  # [batch, seq_len, features]
    out = m(x)
    assert out.shape == (3, 1)
    assert hasattr(m, "attn")


def test_forward_shape_with_attention_disabled_falls_back_to_last_hidden_state():
    m = _make_attn_module(n_features=5, hidden_dim=8, num_layers=2, dropout=0.1,
                           num_heads=2, attention_enabled=False)
    x = torch.randn(3, 10, 5)
    out = m(x)
    assert out.shape == (3, 1)
    assert not hasattr(m, "attn")  # attention submodules must not even be built


def test_disabling_attention_changes_the_forward_path_output():
    torch.manual_seed(0)
    m_on = _make_attn_module(n_features=4, hidden_dim=8, num_layers=1, dropout=0.0,
                              num_heads=2, attention_enabled=True)
    torch.manual_seed(0)
    m_off = _make_attn_module(n_features=4, hidden_dim=8, num_layers=1, dropout=0.0,
                               num_heads=2, attention_enabled=False)
    m_on.eval()
    m_off.eval()
    x = torch.randn(2, 6, 4)
    with torch.no_grad():
        out_on = m_on(x)
        out_off = m_off(x)
    # Regression guard for the enabled-flag fix: before it, this flag was read
    # nowhere and both models behaved identically (attention always built).
    # Different parameter counts change the RNG draw sequence too, so this
    # isn't a clean ablation of the aggregation path alone -- just proof the
    # flag now has an effect at all.
    assert not torch.allclose(out_on, out_off)
