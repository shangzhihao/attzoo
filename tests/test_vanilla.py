"""Function-based test cases for VanillaSelfAttention forward method."""

import torch

from attzoo.vanilla import VanillaSelfAttention
from tests import EPSILON


def test_vanilla_forward_basic_shapes() -> None:
    """Test forward method produces correct output shapes."""
    batch_size, seq_len, d_model = 2, 8, 64
    attention = VanillaSelfAttention(d_model=d_model)
    x = torch.randn(batch_size, seq_len, d_model)

    output, attention_weights = attention(x)

    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (batch_size, seq_len, seq_len)
    assert isinstance(output, torch.Tensor)
    assert isinstance(attention_weights, torch.Tensor)


def test_vanilla_forward_flexible_input_dimensions() -> None:
    """Test forward method with different input dimensions."""
    batch_size, seq_len = 3, 12
    input_dim, d_model = 32, 64

    attention = VanillaSelfAttention(d_model=d_model, input_dim=input_dim)
    x = torch.randn(batch_size, seq_len, input_dim)

    output, attention_weights = attention(x)

    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (batch_size, seq_len, seq_len)


def test_vanilla_forward_with_causal_mask() -> None:
    """Test forward method with causal mask."""
    batch_size, seq_len, d_model = 1, 6, 32
    attention = VanillaSelfAttention(d_model=d_model)
    attention.eval()  # Disable dropout for deterministic behavior

    x = torch.randn(batch_size, seq_len, d_model)
    # Create causal mask (lower triangular)
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()

    _output, attention_weights = attention(x, mask=mask)

    # Check that upper triangular attention weights are masked (near zero)
    upper_triangular = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    masked_positions = attention_weights[0][upper_triangular]
    assert (masked_positions < EPSILON).all()


def test_vanilla_forward_with_padding_mask() -> None:
    """Test forward method with padding mask."""
    batch_size, seq_len, d_model = 2, 5, 32
    attention = VanillaSelfAttention(d_model=d_model)
    attention.eval()

    x = torch.randn(batch_size, seq_len, d_model)
    # Create padding mask (mask out last 2 positions)
    mask = torch.ones(batch_size, seq_len, seq_len).bool()
    mask[:, :, -2:] = False  # Mask out last 2 positions

    _output, attention_weights = attention(x, mask=mask)

    # Check that masked positions have very small attention weights
    assert (attention_weights[:, :, -2:] < EPSILON).all()


def test_vanilla_forward_attention_weights_properties() -> None:
    """Test that attention weights have correct mathematical properties."""
    batch_size, seq_len, d_model = 2, 4, 32
    attention = VanillaSelfAttention(d_model=d_model)
    attention.eval()  # Disable dropout for deterministic behavior

    x = torch.randn(batch_size, seq_len, d_model)
    _output, attention_weights = attention(x)

    # Attention weights should sum to 1 along last dimension
    assert torch.allclose(
        attention_weights.sum(dim=-1), torch.ones(batch_size, seq_len)
    )

    # Attention weights should be non-negative
    assert (attention_weights >= 0).all()

    # Attention weights should be <= 1
    assert (attention_weights <= 1).all()


def test_vanilla_forward_temperature_scaling() -> None:
    """Test forward method with different temperature values."""
    batch_size, seq_len, d_model = 1, 4, 32
    x = torch.randn(batch_size, seq_len, d_model)

    # Test low temperature (sharper attention)
    attention_low = VanillaSelfAttention(d_model=d_model, temperature=0.1)
    attention_low.eval()
    _, weights_low = attention_low(x)

    # Test high temperature (smoother attention)
    attention_high = VanillaSelfAttention(d_model=d_model, temperature=10.0)
    attention_high.eval()
    # Copy weights for fair comparison
    attention_high.load_state_dict(attention_low.state_dict())
    attention_high.temperature = 10.0
    _, weights_high = attention_high(x)

    # Low temperature should produce more concentrated attention
    max_attention_low = weights_low.max(dim=-1)[0]
    max_attention_high = weights_high.max(dim=-1)[0]
    assert (max_attention_low >= max_attention_high).all()


def test_vanilla_forward_deterministic_behavior() -> None:
    """Test that forward method is deterministic when dropout is disabled."""
    batch_size, seq_len, d_model = 1, 3, 16
    attention = VanillaSelfAttention(d_model=d_model)
    attention.eval()  # Disable dropout

    x = torch.randn(batch_size, seq_len, d_model)

    # Run forward pass twice
    output1, weights1 = attention(x)
    output2, weights2 = attention(x)

    # Results should be identical
    assert torch.allclose(output1, output2)
    assert torch.allclose(weights1, weights2)


def test_vanilla_forward_gradient_flow() -> None:
    """Test that gradients flow properly through forward method."""
    batch_size, seq_len, d_model = 1, 3, 16
    attention = VanillaSelfAttention(d_model=d_model)
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

    output, _ = attention(x)
    loss = output.sum()
    loss.backward()

    # Check that input has gradients
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


def test_vanilla_forward_batch_independence() -> None:
    """Test that different batch items are processed independently."""
    seq_len, d_model = 4, 32
    attention = VanillaSelfAttention(d_model=d_model)
    attention.eval()  # Disable dropout for deterministic behavior

    # Create batch with different inputs
    x1 = torch.randn(1, seq_len, d_model)
    x2 = torch.randn(1, seq_len, d_model)
    x_batch = torch.cat([x1, x2], dim=0)

    # Process individually and as batch
    output1, weights1 = attention(x1)
    output2, weights2 = attention(x2)
    output_batch, weights_batch = attention(x_batch)

    # Results should match
    assert torch.allclose(output1, output_batch[0:1], atol=EPSILON)
    assert torch.allclose(output2, output_batch[1:2], atol=EPSILON)
    assert torch.allclose(weights1, weights_batch[0:1], atol=EPSILON)
    assert torch.allclose(weights2, weights_batch[1:2], atol=EPSILON)


def test_vanilla_forward_sequence_length_variation() -> None:
    """Test forward method with different sequence lengths."""
    batch_size, d_model = 1, 32
    attention = VanillaSelfAttention(d_model=d_model)

    for seq_len in [1, 3, 8, 16]:
        x = torch.randn(batch_size, seq_len, d_model)
        output, attention_weights = attention(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (batch_size, seq_len, seq_len)


def test_vanilla_forward_dtype_preservation() -> None:
    """Test that forward method preserves input dtype."""
    batch_size, seq_len, d_model = 1, 4, 32
    attention = VanillaSelfAttention(d_model=d_model)

    # Test with float32
    x_float32 = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)
    output_float32, weights_float32 = attention(x_float32)
    assert output_float32.dtype == torch.float32
    assert weights_float32.dtype == torch.float32

    # Test with float64
    attention_float64 = VanillaSelfAttention(d_model=d_model).to(torch.float64)
    x_float64 = torch.randn(batch_size, seq_len, d_model, dtype=torch.float64)
    output_float64, weights_float64 = attention_float64(x_float64)
    assert output_float64.dtype == torch.float64
    assert weights_float64.dtype == torch.float64


def test_vanilla_forward_edge_cases() -> None:
    """Test forward method with edge cases."""
    d_model = 32
    attention = VanillaSelfAttention(d_model=d_model)
    attention.eval()

    # Test with single element sequence
    x_single = torch.randn(1, 1, d_model)
    output_single, weights_single = attention(x_single)
    assert output_single.shape == (1, 1, d_model)
    assert weights_single.shape == (1, 1, 1)
    assert torch.allclose(weights_single, torch.ones(1, 1, 1))

    # Test with very small values
    x_small = torch.randn(1, 3, d_model) * EPSILON
    output_small, weights_small = attention(x_small)
    assert output_small.shape == (1, 3, d_model)
    assert torch.allclose(weights_small.sum(dim=-1), torch.ones(1, 3))
