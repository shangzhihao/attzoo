"""Function-based test cases for LocalSelfAttention forward method."""

import torch

from attentions.local import LocalSelfAttention
from tests import EPSILON


def test_local_forward_basic_shapes() -> None:
    """Test forward method produces correct output shapes."""
    batch_size, seq_len, d_model = 2, 16, 64
    window_size, num_heads = 8, 8
    attention = LocalSelfAttention(
        d_model=d_model, window_size=window_size, num_heads=num_heads
    )
    x = torch.randn(batch_size, seq_len, d_model)

    output, attention_weights = attention(x)

    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)
    assert isinstance(output, torch.Tensor)
    assert isinstance(attention_weights, torch.Tensor)


def test_local_forward_flexible_input_dimensions() -> None:
    """Test forward method with different input dimensions."""
    batch_size, seq_len = 3, 12
    input_dim, d_model = 48, 96
    window_size, num_heads = 6, 6

    attention = LocalSelfAttention(
        d_model=d_model,
        input_dim=input_dim,
        window_size=window_size,
        num_heads=num_heads,
    )
    x = torch.randn(batch_size, seq_len, input_dim)

    output, attention_weights = attention(x)

    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)


def test_local_forward_windowing_pattern() -> None:
    """Test that forward method produces correct local attention patterns."""
    batch_size, seq_len, d_model = 1, 8, 32
    window_size, num_heads = 4, 4
    attention = LocalSelfAttention(
        d_model=d_model, window_size=window_size, num_heads=num_heads
    )
    attention.eval()  # Disable dropout for deterministic behavior

    x = torch.randn(batch_size, seq_len, d_model)
    _output, attention_weights = attention(x)

    # Check local windowing pattern for each head
    half_window = window_size // 2
    for head in range(num_heads):
        for i in range(seq_len):
            start_idx = max(0, i - half_window)
            end_idx = min(seq_len, i + half_window + 1)

            # Within window should have non-zero attention
            window_weights = attention_weights[0, head, i, start_idx:end_idx]
            assert torch.allclose(window_weights.sum(), torch.tensor(1.0), atol=EPSILON)

            # Outside window should have zero attention
            if start_idx > 0:
                assert torch.allclose(
                    attention_weights[0, head, i, :start_idx], torch.zeros(start_idx)
                )
            if end_idx < seq_len:
                assert torch.allclose(
                    attention_weights[0, head, i, end_idx:],
                    torch.zeros(seq_len - end_idx),
                )


def test_local_forward_different_window_sizes() -> None:
    """Test forward method with different window sizes."""
    batch_size, seq_len, d_model = 1, 12, 32
    num_heads = 4
    x = torch.randn(batch_size, seq_len, d_model)

    for window_size in [2, 4, 6, 8, 12]:
        attention = LocalSelfAttention(
            d_model=d_model, window_size=window_size, num_heads=num_heads
        )
        attention.eval()
        output, attention_weights = attention(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)

        # Check sparsity pattern for each head
        half_window = window_size // 2
        for head in range(num_heads):
            for i in range(seq_len):
                start_idx = max(0, i - half_window)
                end_idx = min(seq_len, i + half_window + 1)

                # Count non-zero attention weights
                non_zero_count = (attention_weights[0, head, i] > EPSILON).sum().item()
                expected_window = end_idx - start_idx
                assert non_zero_count == expected_window


def test_local_forward_with_causal_mask() -> None:
    """Test forward method with local mask combined with causal mask."""
    batch_size, seq_len, d_model = 1, 8, 32
    window_size, num_heads = 6, 4
    attention = LocalSelfAttention(
        d_model=d_model, window_size=window_size, num_heads=num_heads
    )
    attention.eval()

    x = torch.randn(batch_size, seq_len, d_model)
    # Create causal mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()

    _output, attention_weights = attention(x, mask=causal_mask)

    # Check that both local and causal constraints are applied
    half_window = window_size // 2
    for head in range(num_heads):
        for i in range(seq_len):
            for j in range(seq_len):
                # Get expected mask value (both local and causal)
                start_idx = max(0, i - half_window)
                end_idx = min(seq_len, i + half_window + 1)

                local_allowed = start_idx <= j < end_idx
                causal_allowed = j <= i

                if not (local_allowed and causal_allowed):
                    # Should have zero or near-zero attention
                    assert attention_weights[0, head, i, j] < 1e-5


def test_local_forward_with_padding_mask() -> None:
    """Test forward method with padding mask combined with local mask."""
    batch_size, seq_len, d_model = 2, 6, 32
    window_size, num_heads = 4, 4
    attention = LocalSelfAttention(
        d_model=d_model, window_size=window_size, num_heads=num_heads
    )
    attention.eval()

    x = torch.randn(batch_size, seq_len, d_model)
    # Create padding mask (mask out last position)
    padding_mask = torch.ones(batch_size, seq_len, seq_len).bool()
    padding_mask[:, :, -1] = False  # Mask out last position

    _output, attention_weights = attention(x, mask=padding_mask)

    # Check that padded positions are masked for all heads
    for head in range(num_heads):
        assert (attention_weights[:, head, :, -1] < EPSILON).all()


def test_local_forward_window_larger_than_sequence() -> None:
    """Test forward method when window size is larger than sequence length."""
    batch_size, seq_len, d_model = 1, 4, 32
    window_size, num_heads = 10, 4  # Window larger than sequence
    attention = LocalSelfAttention(
        d_model=d_model, window_size=window_size, num_heads=num_heads
    )
    attention.eval()

    x = torch.randn(batch_size, seq_len, d_model)
    _output, attention_weights = attention(x)

    # Should behave like full attention when window >= sequence length
    for head in range(num_heads):
        for i in range(seq_len):
            # All positions should be attended to
            assert torch.allclose(
                attention_weights[0, head, i].sum(), torch.tensor(1.0), atol=EPSILON
            )
            # No position should be masked out
            assert (attention_weights[0, head, i] > EPSILON).all()


def test_local_forward_single_element_window() -> None:
    """Test forward method with window size of 1 (self-attention only)."""
    batch_size, seq_len, d_model = 1, 6, 32
    window_size, num_heads = 1, 4
    attention = LocalSelfAttention(
        d_model=d_model, window_size=window_size, num_heads=num_heads
    )
    attention.eval()

    x = torch.randn(batch_size, seq_len, d_model)
    _output, attention_weights = attention(x)

    # Each position should only attend to itself
    for head in range(num_heads):
        for i in range(seq_len):
            # Only diagonal element should be non-zero
            expected_weights = torch.zeros(seq_len)
            expected_weights[i] = 1.0
            assert torch.allclose(
                attention_weights[0, head, i], expected_weights, atol=EPSILON
            )


def test_local_forward_different_head_counts() -> None:
    """Test forward method with different numbers of heads."""
    batch_size, seq_len, d_model = 2, 8, 64
    window_size = 4
    x = torch.randn(batch_size, seq_len, d_model)

    for num_heads in [1, 2, 4, 8, 16]:
        attention = LocalSelfAttention(
            d_model=d_model, window_size=window_size, num_heads=num_heads
        )
        attention.eval()

        output, attention_weights = attention(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)
        assert attention.d_head == d_model // num_heads


def test_local_forward_temperature_scaling() -> None:
    """Test forward method with different temperature values."""
    batch_size, seq_len, d_model = 1, 6, 32
    window_size, num_heads = 4, 4
    x = torch.randn(batch_size, seq_len, d_model)

    # Test low temperature (sharper attention)
    attention_low = LocalSelfAttention(
        d_model=d_model, window_size=window_size, num_heads=num_heads, temperature=0.1
    )
    attention_low.eval()
    _, weights_low = attention_low(x)

    # Test high temperature (smoother attention)
    attention_high = LocalSelfAttention(
        d_model=d_model, window_size=window_size, num_heads=num_heads, temperature=10.0
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


def test_local_forward_deterministic_behavior() -> None:
    """Test that forward method is deterministic when dropout is disabled."""
    batch_size, seq_len, d_model = 1, 8, 32
    window_size, num_heads = 4, 4
    attention = LocalSelfAttention(
        d_model=d_model, window_size=window_size, num_heads=num_heads
    )
    attention.eval()  # Disable dropout

    x = torch.randn(batch_size, seq_len, d_model)

    # Run forward pass twice
    output1, weights1 = attention(x)
    output2, weights2 = attention(x)

    # Results should be identical
    assert torch.allclose(output1, output2)
    assert torch.allclose(weights1, weights2)


def test_local_forward_gradient_flow() -> None:
    """Test that gradients flow properly through forward method."""
    batch_size, seq_len, d_model = 1, 6, 32
    window_size, num_heads = 4, 4
    attention = LocalSelfAttention(
        d_model=d_model, window_size=window_size, num_heads=num_heads
    )
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

    output, _ = attention(x)
    loss = output.sum()
    loss.backward()

    # Check that input has gradients
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


def test_local_forward_sequence_length_variation() -> None:
    """Test forward method with different sequence lengths."""
    batch_size, d_model = 1, 32
    window_size, num_heads = 6, 4
    attention = LocalSelfAttention(
        d_model=d_model, window_size=window_size, num_heads=num_heads
    )

    for seq_len in [1, 4, 8, 12, 16]:
        x = torch.randn(batch_size, seq_len, d_model)
        output, attention_weights = attention(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)


def test_local_forward_stored_attention_weights() -> None:
    """Test that stored attention weights are correctly computed."""
    batch_size, seq_len, d_model = 1, 8, 32
    window_size, num_heads = 4, 4
    attention = LocalSelfAttention(
        d_model=d_model, window_size=window_size, num_heads=num_heads
    )
    attention.eval()

    x = torch.randn(batch_size, seq_len, d_model)
    _output, attention_weights = attention(x)

    # Get stored attention weights (should be averaged across heads)
    stored_weights = attention.get_attention_weights()

    # Should match manual average
    expected_weights = attention_weights.mean(dim=1)
    assert torch.allclose(stored_weights, expected_weights, atol=EPSILON)
    assert stored_weights.shape == (batch_size, seq_len, seq_len)


def test_local_forward_attention_sparsity() -> None:
    """Test that local attention produces sparse attention patterns."""
    batch_size, seq_len, d_model = 1, 16, 32
    window_size, num_heads = 6, 4
    attention = LocalSelfAttention(
        d_model=d_model, window_size=window_size, num_heads=num_heads
    )
    attention.eval()

    x = torch.randn(batch_size, seq_len, d_model)
    _output, attention_weights = attention(x)

    # Count total non-zero attention weights
    total_elements = batch_size * num_heads * seq_len * seq_len
    non_zero_elements = (attention_weights > EPSILON).sum().item()
    sparsity_ratio = non_zero_elements / total_elements

    # Local attention should be significantly more sparse than full attention
    # Expected sparsity depends on window size and sequence length
    expected_max_sparsity = window_size / seq_len  # Rough estimate
    assert sparsity_ratio <= expected_max_sparsity * 1.1  # Allow some tolerance


def test_local_forward_complexity_reduction() -> None:
    """Test that local attention reduces computational complexity."""
    d_model = 32
    window_size, num_heads = 8, 4

    # Test with different sequence lengths
    for seq_len in [16, 32, 64]:
        attention = LocalSelfAttention(
            d_model=d_model, window_size=window_size, num_heads=num_heads
        )
        attention.eval()

        x = torch.randn(1, seq_len, d_model)
        _output, attention_weights = attention(x)

        # Count effective attention computations per position
        half_window = window_size // 2

        # Each position should attend to at most the window size
        # The actual window for position i is [max(0, i-half_window), min(seq_len, i+half_window+1))
        for head in range(num_heads):
            for i in range(seq_len):
                start_idx = max(0, i - half_window)
                end_idx = min(seq_len, i + half_window + 1)
                expected_window_size = end_idx - start_idx

                non_zero_count = (attention_weights[0, head, i] > EPSILON).sum().item()
                assert non_zero_count == expected_window_size
