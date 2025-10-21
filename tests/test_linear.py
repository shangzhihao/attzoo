"""Function-based test cases for LinearSelfAttention forward method."""

import torch

from attentions.linear import LinearSelfAttention


def test_linear_forward_basic_shapes() -> None:
    """Test forward method produces correct output shapes."""
    batch_size, seq_len, d_model = 2, 8, 64
    attention = LinearSelfAttention(d_model=d_model)
    x = torch.randn(batch_size, seq_len, d_model)

    output, attention_weights = attention(x)

    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (batch_size, seq_len, seq_len)
    assert isinstance(output, torch.Tensor)
    assert isinstance(attention_weights, torch.Tensor)


def test_linear_forward_flexible_input_dimensions() -> None:
    """Test forward method with different input dimensions."""
    batch_size, seq_len = 3, 12
    input_dim, d_model = 32, 64

    attention = LinearSelfAttention(d_model=d_model, input_dim=input_dim)
    x = torch.randn(batch_size, seq_len, input_dim)

    output, attention_weights = attention(x)

    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (batch_size, seq_len, seq_len)


def test_linear_forward_custom_feature_dimensions() -> None:
    """Test forward method with custom feature dimensions."""
    batch_size, seq_len, d_model = 2, 6, 64
    feature_dim = 32

    attention = LinearSelfAttention(d_model=d_model, feature_dim=feature_dim)
    x = torch.randn(batch_size, seq_len, d_model)

    output, attention_weights = attention(x)

    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (batch_size, seq_len, seq_len)


def test_linear_forward_with_padding_mask() -> None:
    """Test forward method with padding mask."""
    batch_size, seq_len, d_model = 2, 5, 32
    attention = LinearSelfAttention(d_model=d_model)
    attention.eval()

    x = torch.randn(batch_size, seq_len, d_model)
    # Create padding mask (mask out last 2 positions)
    mask = torch.ones(batch_size, seq_len).bool()
    mask[:, -2:] = False  # Mask out last 2 positions

    output, attention_weights = attention(x, mask=mask)

    # Output should have reduced magnitude at masked positions
    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (batch_size, seq_len, seq_len)


def test_linear_forward_with_attention_mask() -> None:
    """Test forward method with attention mask."""
    batch_size, seq_len, d_model = 1, 4, 32
    attention = LinearSelfAttention(d_model=d_model)
    attention.eval()

    x = torch.randn(batch_size, seq_len, d_model)
    # Create attention mask
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()

    output, attention_weights = attention(x, mask=mask)

    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (batch_size, seq_len, seq_len)


def test_linear_forward_attention_weights_properties() -> None:
    """Test that attention weights have reasonable properties."""
    batch_size, seq_len, d_model = 2, 4, 32
    attention = LinearSelfAttention(d_model=d_model)
    attention.eval()  # Disable dropout for deterministic behavior

    x = torch.randn(batch_size, seq_len, d_model)
    _output, attention_weights = attention(x)

    # Attention weights should be non-negative (approximately)
    assert (attention_weights >= -1e-6).all()

    # Attention weights should sum to approximately 1 (linear attention is approximate)
    weight_sums = attention_weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=0.1)


def test_linear_forward_deterministic_behavior() -> None:
    """Test that forward method is deterministic when dropout is disabled."""
    batch_size, seq_len, d_model = 1, 3, 16
    attention = LinearSelfAttention(d_model=d_model)
    attention.eval()  # Disable dropout

    x = torch.randn(batch_size, seq_len, d_model)

    # Run forward pass twice
    output1, weights1 = attention(x)
    output2, weights2 = attention(x)

    # Results should be identical
    assert torch.allclose(output1, output2)
    assert torch.allclose(weights1, weights2)


def test_linear_forward_batch_independence() -> None:
    """Test that different batch items are processed independently."""
    seq_len, d_model = 4, 32
    attention = LinearSelfAttention(d_model=d_model)
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
    assert torch.allclose(output1, output_batch[0:1], atol=1e-6)
    assert torch.allclose(output2, output_batch[1:2], atol=1e-6)
    assert torch.allclose(weights1, weights_batch[0:1], atol=1e-6)
    assert torch.allclose(weights2, weights_batch[1:2], atol=1e-6)
