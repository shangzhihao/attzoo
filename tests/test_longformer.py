"""Tests for LongformerSelfAttention (sliding window + global tokens)."""

import torch

from attzoo.longformer import LongformerSelfAttention
from tests import EPSILON


def _longformer_attention_with_global_token() -> tuple[torch.Tensor, int, int]:
    """Compute attention weights for a setup with a single global token."""
    batch_size, seq_len, d_model = 1, 10, 40
    num_heads, window_size = 4, 4
    x = torch.randn(batch_size, seq_len, d_model)

    global_index = 6
    global_mask = torch.zeros(seq_len, dtype=torch.bool)
    global_mask[global_index] = True

    attn = LongformerSelfAttention(
        d_model=d_model, num_heads=num_heads, window_size=window_size
    )
    attn.eval()
    with torch.no_grad():
        _, weights = attn(x, global_attention=global_mask)

    return weights, global_index, window_size


def test_longformer_forward_basic_shapes() -> None:
    batch_size, seq_len, d_model = 2, 12, 48
    num_heads, window_size = 4, 6
    x = torch.randn(batch_size, seq_len, d_model)

    attn = LongformerSelfAttention(
        d_model=d_model, num_heads=num_heads, window_size=window_size
    )
    out, weights = attn(x)

    assert out.shape == (batch_size, seq_len, d_model)
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)


def test_longformer_window_pattern_respects_local_mask() -> None:
    """Ensure non-global tokens attend only within their sliding windows."""
    w, g_idx, window_size = _longformer_attention_with_global_token()
    num_heads = w.shape[1]
    seq_len = w.shape[2]

    half = window_size // 2
    indices = torch.arange(seq_len)
    col_offsets = indices.unsqueeze(0) - indices.unsqueeze(1)
    within_window = col_offsets.abs() <= half
    within_window[:, g_idx] = True  # global column always allowed
    within_window[g_idx] = True  # skip expectations for global row
    expected_zero_mask = ~within_window
    expected_zero_mask[g_idx] = False

    for head in range(num_heads):
        selected = w[0, head][expected_zero_mask]
        assert (selected < EPSILON).all()


def test_longformer_global_token_attends_everywhere() -> None:
    """Verify probability mass distribution for the designated global token."""
    w, g_idx, _ = _longformer_attention_with_global_token()
    num_heads = w.shape[1]

    for head in range(num_heads):
        assert (w[0, head, g_idx, :] > EPSILON).all()
        assert (w[0, head, :, g_idx] > EPSILON).all()


def test_longformer_with_causal_mask_and_globals() -> None:
    """Causal base mask should still constrain future positions even for globals."""
    batch_size, seq_len, d_model = 1, 8, 32
    num_heads, window_size = 2, 4
    x = torch.randn(batch_size, seq_len, d_model)

    g = 5  # global position
    global_mask = torch.zeros(seq_len, dtype=torch.bool)
    global_mask[g] = True

    causal = torch.tril(torch.ones(seq_len, seq_len)).bool()

    attn = LongformerSelfAttention(
        d_model=d_model, num_heads=num_heads, window_size=window_size
    )
    attn.eval()
    _, w = attn(x, mask=causal, global_attention=global_mask)

    # For i < g, attending to future global position g must be masked by causal
    for head in range(num_heads):
        for i in range(g):
            assert w[0, head, i, g] < EPSILON

    # For i >= g, attending to global position g is allowed
    for head in range(num_heads):
        for i in range(g, seq_len):
            assert w[0, head, i, g] > EPSILON


def test_longformer_additive_mask_support() -> None:
    """Additive mask (-inf for masked) should zero-out those positions."""
    batch_size, seq_len, d_model = 1, 7, 28
    num_heads, window_size = 2, 4
    x = torch.randn(batch_size, seq_len, d_model)

    # Mask the last column for everyone via additive mask
    additive = torch.zeros(seq_len, seq_len)
    additive[:, -1] = float("-inf")

    attn = LongformerSelfAttention(
        d_model=d_model, num_heads=num_heads, window_size=window_size
    )
    attn.eval()
    _, w = attn(x, mask=additive)

    # Last column must be zero for all heads and all i
    for head in range(num_heads):
        assert (w[0, head, :, -1] < EPSILON).all()


def test_longformer_window_larger_than_sequence_behaves_full() -> None:
    batch_size, seq_len, d_model = 1, 5, 20
    num_heads, window_size = 2, 64  # window >> seq_len
    x = torch.randn(batch_size, seq_len, d_model)

    attn = LongformerSelfAttention(
        d_model=d_model, num_heads=num_heads, window_size=window_size
    )
    attn.eval()
    _, w = attn(x)

    # All entries should be unmasked (positive after softmax)
    assert (w > EPSILON).all()


def test_longformer_single_element_window_self_only() -> None:
    batch_size, seq_len, d_model = 1, 6, 24
    num_heads, window_size = 3, 1
    x = torch.randn(batch_size, seq_len, d_model)

    attn = LongformerSelfAttention(
        d_model=d_model, num_heads=num_heads, window_size=window_size
    )
    attn.eval()
    _, w = attn(x)

    # Only diagonal should be non-zero when no globals
    for head in range(num_heads):
        for i in range(seq_len):
            expected = torch.zeros(seq_len)
            expected[i] = 1.0
            assert torch.allclose(w[0, head, i], expected, atol=1e-6)


def test_longformer_temperature_scaling() -> None:
    batch_size, seq_len, d_model = 1, 8, 32
    num_heads, window_size = 4, 4
    x = torch.randn(batch_size, seq_len, d_model)

    low = LongformerSelfAttention(
        d_model=d_model, num_heads=num_heads, window_size=window_size, temperature=0.1
    )
    high = LongformerSelfAttention(
        d_model=d_model, num_heads=num_heads, window_size=window_size, temperature=10.0
    )
    low.eval()
    high.eval()

    # Align parameters for fair comparison
    with torch.no_grad():
        for p_low, p_high in zip(low.parameters(), high.parameters(), strict=False):
            p_high.copy_(p_low)

    _, w_low = low(x)
    _, w_high = high(x)

    # Low temperature should yield sharper (higher max) attention per head
    for head in range(num_heads):
        assert w_low[0, head].max() >= w_high[0, head].max()


def test_longformer_rope_toggle_and_stored_weights() -> None:
    batch_size, seq_len, d_model = 1, 10, 40
    num_heads, window_size = 5, 6
    x = torch.randn(batch_size, seq_len, d_model)

    attn = LongformerSelfAttention(
        d_model=d_model, num_heads=num_heads, window_size=window_size, rope=True
    )
    attn.eval()
    out, w = attn(x)

    # Shapes
    assert out.shape == (batch_size, seq_len, d_model)
    assert w.shape == (batch_size, num_heads, seq_len, seq_len)

    # Stored weights are mean over heads
    stored = attn.get_attention_weights()
    assert torch.allclose(stored, w.mean(dim=1), atol=1e-6)
    assert stored.shape == (batch_size, seq_len, seq_len)


def test_longformer_global_mask_batch_and_vector_forms() -> None:
    batch_size, seq_len, d_model = 2, 9, 36
    num_heads, window_size = 3, 4
    x = torch.randn(batch_size, seq_len, d_model)

    shared_global_index = 2
    batch_only_global_index = 7

    # Vector form (same globals for all in batch)
    v = torch.zeros(seq_len, dtype=torch.bool)
    v[shared_global_index] = True

    # Batch form (different for sample 0 and 1)
    b = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    b[0, shared_global_index] = True
    b[1, batch_only_global_index] = True

    attn = LongformerSelfAttention(
        d_model=d_model, num_heads=num_heads, window_size=window_size
    )
    attn.eval()

    # Vector form
    _, w_vec = attn(x, global_attention=v)

    # Batch form
    _, w_batch = attn(x, global_attention=b)

    # For sample 0, column 2 must be >0 everywhere in both cases
    for head in range(num_heads):
        assert (w_vec[0, head, :, shared_global_index] > EPSILON).all()
        assert (w_batch[0, head, :, shared_global_index] > EPSILON).all()

    # For sample 1, vector form has no global at 7; batch form does
    for head in range(num_heads):
        # Batch form should enable col 7 for all i
        assert (w_batch[1, head, :, batch_only_global_index] > EPSILON).all()
        # Vector form: column 7 behaves as local-only for non-window positions
        # Find an i far from 7
        i_far = 0
        half = window_size // 2
        outside_window = not (
            max(0, i_far - half)
            <= batch_only_global_index
            < min(seq_len, i_far + half + 1)
        )
        assert outside_window
        assert w_vec[1, head, i_far, batch_only_global_index] < EPSILON
