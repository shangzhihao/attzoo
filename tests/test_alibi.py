"""Tests for ALiBi (Attention with Linear Biases) Self-Attention."""

from typing import cast

import pytest
import torch

from attentions.alibi import AlibiSelfAttention


def test_alibi_forward_basic_shapes() -> None:
    """Test AlibiSelfAttention forward pass output shapes."""
    batch_size, seq_len, d_model = 2, 8, 64
    num_heads = 4

    attention = AlibiSelfAttention(d_model=d_model, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, d_model)

    output, attention_weights = attention(x)

    # Check output shapes
    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)

    # Check stored attention weights (averaged across heads)
    stored_weights = attention.get_attention_weights()
    assert stored_weights.shape == (batch_size, seq_len, seq_len)


def test_alibi_forward_flexible_input_dimensions() -> None:
    """Test AlibiSelfAttention with different input dimensions."""
    batch_size, seq_len, input_dim, d_model = 2, 6, 128, 64
    num_heads = 8

    attention = AlibiSelfAttention(
        d_model=d_model, num_heads=num_heads, input_dim=input_dim
    )
    x = torch.randn(batch_size, seq_len, input_dim)

    output, attention_weights = attention(x)

    # Output should have d_model dimensions
    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)


def test_alibi_slopes_computation() -> None:
    """Test ALiBi slopes computation for different numbers of heads."""
    # Test power of 2 heads
    attention_4 = AlibiSelfAttention(d_model=32, num_heads=4)
    slopes_4 = cast(torch.Tensor, attention_4.slopes)
    assert slopes_4.shape == (4,)
    assert slopes_4.dtype == torch.float32

    # Test non-power of 2 heads
    attention_6 = AlibiSelfAttention(d_model=48, num_heads=6)
    slopes_6 = cast(torch.Tensor, attention_6.slopes)
    assert slopes_6.shape == (6,)

    # For power of 2, slopes should be monotonically decreasing
    assert torch.all(slopes_4[:-1] >= slopes_4[1:])

    # For non-power of 2, slopes pattern may vary but should be valid
    assert torch.all(slopes_6 > 0)  # All slopes should be positive


def test_alibi_bias_creation() -> None:
    """Test ALiBi bias matrix creation."""
    d_model, num_heads, max_seq_len = 32, 4, 16
    attention = AlibiSelfAttention(
        d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len
    )

    bias = cast(torch.Tensor, attention.alibi_bias)
    slopes = cast(torch.Tensor, attention.slopes)

    # Check bias matrix shape
    assert bias.shape == (num_heads, max_seq_len, max_seq_len)

    # Check that bias is zero on diagonal
    for head in range(num_heads):
        diagonal = torch.diag(bias[head])
        assert torch.allclose(diagonal, torch.zeros_like(diagonal))

    # Check that bias is symmetric and negative
    for head in range(num_heads):
        bias_matrix = bias[head]
        assert torch.allclose(bias_matrix, bias_matrix.T)  # Symmetric
        assert torch.all(bias_matrix <= 0)  # Non-positive

    # Check distance-based bias pattern
    for head in range(num_heads):
        # Position (0, 1) should have bias = -slope * 1
        expected_bias_01 = -slopes[head] * 1
        assert torch.allclose(bias[head, 0, 1], expected_bias_01, atol=1e-6)

        # Position (0, 3) should have bias = -slope * 3
        expected_bias_03 = -slopes[head] * 3
        assert torch.allclose(bias[head, 0, 3], expected_bias_03, atol=1e-6)


def test_alibi_forward_different_head_counts() -> None:
    """Test ALiBi attention with different numbers of heads."""
    batch_size, seq_len, d_model = 2, 6, 64
    x = torch.randn(batch_size, seq_len, d_model)

    for num_heads in [1, 2, 4, 8, 16]:
        attention = AlibiSelfAttention(d_model=d_model, num_heads=num_heads)
        output, attention_weights = attention(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)
        assert attention.d_head == d_model // num_heads


def test_alibi_forward_with_causal_mask() -> None:
    """Test ALiBi attention with causal mask."""
    batch_size, seq_len, d_model = 1, 6, 32
    num_heads = 4
    attention = AlibiSelfAttention(d_model=d_model, num_heads=num_heads)
    attention.eval()  # Disable dropout for deterministic behavior

    x = torch.randn(batch_size, seq_len, d_model)
    # Create causal mask (lower triangular)
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()

    _output, attention_weights = attention(x, mask=mask)

    # Check that upper triangular attention weights are masked for all heads
    upper_triangular = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    for head in range(num_heads):
        masked_positions = attention_weights[0, head][upper_triangular]
        assert (masked_positions < 1e-6).all()


def test_alibi_forward_with_batch_mask() -> None:
    """Test ALiBi attention with batch-specific masks."""
    batch_size, seq_len, d_model = 2, 5, 32
    num_heads = 4
    attention = AlibiSelfAttention(d_model=d_model, num_heads=num_heads)
    attention.eval()

    x = torch.randn(batch_size, seq_len, d_model)
    # Create batch-specific masks
    mask = torch.ones(batch_size, seq_len, seq_len).bool()
    mask[0, :, -1] = False  # Mask last position for first batch item
    mask[1, :, :2] = False  # Mask first two positions for second batch item

    _output, attention_weights = attention(x, mask=mask)

    # Check masking for all heads
    for head in range(num_heads):
        assert (attention_weights[0, head, :, -1] < 1e-6).all()  # Last position masked
        assert (
            attention_weights[1, head, :, :2] < 1e-6
        ).all()  # First two positions masked


def test_alibi_forward_long_sequences() -> None:
    """Test ALiBi attention with sequences longer than max_seq_len."""
    d_model, num_heads = 32, 4
    max_seq_len = 8
    long_seq_len = 12

    attention = AlibiSelfAttention(
        d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len
    )

    # Test with sequence longer than pre-computed bias
    x = torch.randn(1, long_seq_len, d_model)
    output, attention_weights = attention(x)

    assert output.shape == (1, long_seq_len, d_model)
    assert attention_weights.shape == (1, num_heads, long_seq_len, long_seq_len)


def test_alibi_forward_position_bias_effect() -> None:
    """Test that ALiBi creates position-dependent attention patterns."""
    seq_len, d_model, num_heads = 8, 32, 2
    attention = AlibiSelfAttention(d_model=d_model, num_heads=num_heads)
    attention.eval()

    # Create input where all positions have the same content
    x = torch.ones(1, seq_len, d_model)
    _output, attention_weights = attention(x)

    # For identical inputs, attention should still vary by position due to ALiBi bias
    # Check that attention weights are different for different positions
    for head in range(num_heads):
        weights = attention_weights[0, head]

        # Attention from first position should decrease with distance
        first_row = weights[0]
        # Due to ALiBi bias, closer positions should generally have higher attention
        # (though this can be affected by the specific slopes and identical inputs)
        assert not torch.allclose(first_row, torch.ones_like(first_row) / seq_len)


def test_alibi_forward_temperature_scaling() -> None:
    """Test ALiBi attention temperature scaling effects."""
    batch_size, seq_len, d_model = 1, 4, 16
    num_heads = 2

    x = torch.randn(batch_size, seq_len, d_model)

    # Low temperature (sharper attention)
    attention_low = AlibiSelfAttention(
        d_model=d_model, num_heads=num_heads, temperature=0.1
    )
    attention_low.eval()

    # High temperature (smoother attention)
    attention_high = AlibiSelfAttention(
        d_model=d_model, num_heads=num_heads, temperature=10.0
    )
    attention_high.eval()

    # Copy weights to ensure same initialization
    with torch.no_grad():
        for low_param, high_param in zip(
            attention_low.parameters(), attention_high.parameters(), strict=False
        ):
            high_param.copy_(low_param)

    _, weights_low = attention_low(x)
    _, weights_high = attention_high(x)

    # Low temperature should have higher maximum attention weights
    max_low = weights_low.max()
    max_high = weights_high.max()
    assert max_low > max_high


def test_alibi_forward_initialization_errors() -> None:
    """Test ALiBi attention initialization error conditions."""
    # d_model not divisible by num_heads
    with pytest.raises(ValueError, match="d_model must be divisible by num_heads"):
        AlibiSelfAttention(d_model=31, num_heads=4)
