"""Function-based test cases for MultiHeadSelfAttention forward method."""

import torch

from attzoo.mhsa import MultiHeadSelfAttention
from tests import EPSILON


def test_mhsa_forward_basic_shapes() -> None:
    """Test forward method produces correct output shapes."""
    batch_size, seq_len, d_model = 2, 8, 64
    num_heads = 8
    attention = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, d_model)

    output, attention_weights = attention(x)

    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)
    assert isinstance(output, torch.Tensor)
    assert isinstance(attention_weights, torch.Tensor)


def test_mhsa_forward_flexible_input_dimensions() -> None:
    """Test forward method with different input dimensions."""
    batch_size, seq_len = 3, 10
    input_dim, d_model = 48, 96
    num_heads = 6

    attention = MultiHeadSelfAttention(
        d_model=d_model, num_heads=num_heads, input_dim=input_dim
    )
    x = torch.randn(batch_size, seq_len, input_dim)

    output, attention_weights = attention(x)

    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)


def test_mhsa_forward_different_head_counts() -> None:
    """Test forward method with different numbers of heads."""
    batch_size, seq_len, d_model = 2, 6, 64
    x = torch.randn(batch_size, seq_len, d_model)

    for num_heads in [1, 2, 4, 8, 16]:
        attention = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
        output, attention_weights = attention(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)
        assert attention.d_head == d_model // num_heads


def test_mhsa_forward_with_causal_mask() -> None:
    """Test forward method with causal mask."""
    batch_size, seq_len, d_model = 1, 5, 32
    num_heads = 4
    attention = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
    attention.eval()  # Disable dropout for deterministic behavior

    x = torch.randn(batch_size, seq_len, d_model)
    # Create causal mask (lower triangular)
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()

    _output, attention_weights = attention(x, mask=mask)

    # Check that upper triangular attention weights are masked for all heads
    upper_triangular = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    for head in range(num_heads):
        masked_positions = attention_weights[0, head][upper_triangular]
        assert (masked_positions < EPSILON).all()


def test_mhsa_forward_with_batch_mask() -> None:
    """Test forward method with batch-specific masks."""
    batch_size, seq_len, d_model = 2, 4, 32
    num_heads = 4
    attention = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
    attention.eval()

    x = torch.randn(batch_size, seq_len, d_model)
    # Create batch-specific masks
    mask = torch.ones(batch_size, seq_len, seq_len).bool()
    mask[0, :, -1] = False  # Mask last position for first batch item
    mask[1, :, :2] = False  # Mask first two positions for second batch item

    _output, attention_weights = attention(x, mask=mask)

    # Check masking for all heads
    for head in range(num_heads):
        assert (
            attention_weights[0, head, :, -1] < EPSILON
        ).all()  # Last position masked
        assert (
            attention_weights[1, head, :, :2] < EPSILON
        ).all()  # First two positions masked


def test_mhsa_forward_attention_weights_per_head() -> None:
    """Test that attention weights have correct properties for each head."""
    batch_size, seq_len, d_model = 2, 4, 32
    num_heads = 4
    attention = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
    attention.eval()  # Disable dropout for deterministic behavior

    x = torch.randn(batch_size, seq_len, d_model)
    _output, attention_weights = attention(x)

    # Check properties for each head
    for head in range(num_heads):
        head_weights = attention_weights[:, head, :, :]

        # Attention weights should sum to 1 along last dimension
        assert torch.allclose(
            head_weights.sum(dim=-1), torch.ones(batch_size, seq_len), atol=EPSILON
        )

        # Attention weights should be non-negative
        assert (head_weights >= 0).all()

        # Attention weights should be <= 1
        assert (head_weights <= 1).all()


def test_mhsa_forward_head_independence() -> None:
    """Test that different heads produce different attention patterns."""
    batch_size, seq_len, d_model = 1, 6, 48
    num_heads = 8
    attention = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)

    x = torch.randn(batch_size, seq_len, d_model)
    _output, attention_weights = attention(x)

    # Check that not all heads have identical attention patterns
    head_patterns = attention_weights[0]  # Shape: [num_heads, seq_len, seq_len]

    # Compare pairs of heads
    different_patterns = 0
    for i in range(num_heads):
        for j in range(i + 1, num_heads):
            if not torch.allclose(head_patterns[i], head_patterns[j], atol=1e-3):
                different_patterns += 1

    # Most heads should have different patterns
    assert different_patterns > 0


def test_mhsa_forward_temperature_scaling() -> None:
    """Test forward method with different temperature values."""
    batch_size, seq_len, d_model = 1, 4, 32
    num_heads = 4
    x = torch.randn(batch_size, seq_len, d_model)

    # Test low temperature (sharper attention)
    attention_low = MultiHeadSelfAttention(
        d_model=d_model, num_heads=num_heads, temperature=0.1
    )
    attention_low.eval()
    _, weights_low = attention_low(x)

    # Test high temperature (smoother attention)
    attention_high = MultiHeadSelfAttention(
        d_model=d_model, num_heads=num_heads, temperature=10.0
    )
    attention_high.eval()
    # Copy weights for fair comparison
    attention_high.load_state_dict(attention_low.state_dict())
    attention_high.temperature = 10.0
    _, weights_high = attention_high(x)

    # Low temperature should produce more concentrated attention for each head
    for head in range(num_heads):
        max_attention_low = weights_low[0, head].max(dim=-1)[0]
        max_attention_high = weights_high[0, head].max(dim=-1)[0]
        assert (max_attention_low >= max_attention_high).all()


def test_mhsa_forward_deterministic_behavior() -> None:
    """Test that forward method is deterministic when dropout is disabled."""
    batch_size, seq_len, d_model = 1, 3, 24
    num_heads = 6
    attention = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
    attention.eval()  # Disable dropout

    x = torch.randn(batch_size, seq_len, d_model)

    # Run forward pass twice
    output1, weights1 = attention(x)
    output2, weights2 = attention(x)

    # Results should be identical
    assert torch.allclose(output1, output2)
    assert torch.allclose(weights1, weights2)


def test_mhsa_forward_gradient_flow() -> None:
    """Test that gradients flow properly through forward method."""
    batch_size, seq_len, d_model = 1, 3, 24
    num_heads = 6
    attention = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

    output, _ = attention(x)
    loss = output.sum()
    loss.backward()

    # Check that input has gradients
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


def test_mhsa_forward_mask_broadcasting() -> None:
    """Test that masks are properly broadcasted across heads."""
    batch_size, seq_len, d_model = 2, 4, 32
    num_heads = 8
    attention = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
    attention.eval()

    x = torch.randn(batch_size, seq_len, d_model)
    # Create 2D mask that should be broadcasted
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()

    _output, attention_weights = attention(x, mask=mask)

    # Check that mask is applied to all heads and batches
    upper_triangular = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    for batch in range(batch_size):
        for head in range(num_heads):
            masked_positions = attention_weights[batch, head][upper_triangular]
            assert (masked_positions < EPSILON).all()


def test_mhsa_forward_sequence_length_variation() -> None:
    """Test forward method with different sequence lengths."""
    batch_size, d_model = 1, 32
    num_heads = 4
    attention = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)

    for seq_len in [1, 3, 8, 16]:
        x = torch.randn(batch_size, seq_len, d_model)
        output, attention_weights = attention(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)


def test_mhsa_forward_edge_cases() -> None:
    """Test forward method with edge cases."""
    d_model = 32
    num_heads = 4
    attention = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
    attention.eval()

    # Test with single element sequence
    x_single = torch.randn(1, 1, d_model)
    output_single, weights_single = attention(x_single)
    assert output_single.shape == (1, 1, d_model)
    assert weights_single.shape == (1, num_heads, 1, 1)
    assert torch.allclose(weights_single, torch.ones(1, num_heads, 1, 1))

    # Test with very small values
    x_small = torch.randn(1, 3, d_model) * EPSILON
    output_small, weights_small = attention(x_small)
    assert output_small.shape == (1, 3, d_model)
    for head in range(num_heads):
        assert torch.allclose(
            weights_small[0, head].sum(dim=-1), torch.ones(3), atol=EPSILON
        )


def test_mhsa_forward_head_dimension_calculation() -> None:
    """Test that head dimensions are calculated correctly."""
    d_model = 64

    for num_heads in [1, 2, 4, 8, 16]:
        attention = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
        expected_d_head = d_model // num_heads
        assert attention.d_head == expected_d_head
        assert attention.num_heads * attention.d_head == d_model


def test_mhsa_forward_stored_attention_weights() -> None:
    """Test that stored attention weights are correctly computed."""
    batch_size, seq_len, d_model = 1, 4, 32
    num_heads = 8
    attention = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
    attention.eval()

    x = torch.randn(batch_size, seq_len, d_model)
    _output, attention_weights = attention(x)

    # Get stored attention weights (should be averaged across heads)
    stored_weights = attention.get_attention_weights()

    # Should match manual average
    expected_weights = attention_weights.mean(dim=1)
    assert torch.allclose(stored_weights, expected_weights, atol=EPSILON)
    assert stored_weights.shape == (batch_size, seq_len, seq_len)
