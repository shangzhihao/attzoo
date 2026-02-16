import torch

from attzoo.dilation import DilatedSelfAttention
from attzoo.masks import create_dilated_mask


def test_dilated_self_attention_forward_shapes() -> None:
    """Test DilatedSelfAttention forward pass output shapes."""
    batch_size, seq_len, d_model = 2, 10, 64
    num_heads = 4

    attention = DilatedSelfAttention(
        d_model=d_model, num_heads=num_heads, dilation_rate=2
    )
    x = torch.randn(batch_size, seq_len, d_model)

    output, attention_weights = attention(x)

    # Check output shapes
    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)

    # Check stored attention weights (averaged across heads)
    stored_weights = attention.get_attention_weights()
    assert stored_weights.shape == (batch_size, seq_len, seq_len)


def test_dilated_self_attention_forward_single_head() -> None:
    """Test DilatedSelfAttention with single head."""
    batch_size, seq_len, d_model = 2, 8, 32

    attention = DilatedSelfAttention(d_model=d_model, num_heads=1, dilation_rate=3)
    x = torch.randn(batch_size, seq_len, d_model)

    output, attention_weights = attention(x)

    # Check output shapes
    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (batch_size, 1, seq_len, seq_len)

    # Check stored attention weights
    stored_weights = attention.get_attention_weights()
    assert stored_weights.shape == (batch_size, seq_len, seq_len)


def test_dilated_self_attention_forward_flexible_input() -> None:
    """Test DilatedSelfAttention with flexible input dimensions."""
    batch_size, seq_len, input_dim, d_model = 2, 6, 128, 64

    attention = DilatedSelfAttention(
        d_model=d_model, dilation_rate=2, num_heads=4, input_dim=input_dim
    )
    x = torch.randn(batch_size, seq_len, input_dim)

    output, attention_weights = attention(x)

    # Output should have d_model dimensions
    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (batch_size, 4, seq_len, seq_len)


def test_dilated_self_attention_dilation_pattern() -> None:
    """Test DilatedSelfAttention dilation pattern creation."""
    seq_len = 8
    dilation_rate = 2

    mask = create_dilated_mask(seq_len, dilation_rate, torch.device("cpu"))

    # Check mask properties
    assert mask.shape == (seq_len, seq_len)
    assert mask.dtype == torch.bool

    # Check that diagonal is always True (self-attention)
    assert torch.diag(mask).all()

    # Check dilation pattern for a specific position
    # For position 2 with dilation_rate=2, should attend to: 0, 2, 4, 6
    expected_pattern = torch.tensor(
        [True, False, True, False, True, False, True, False]
    )
    assert torch.equal(mask[2], expected_pattern)

    # For position 3 with dilation_rate=2, should attend to: 1, 3, 5, 7
    expected_pattern = torch.tensor(
        [False, True, False, True, False, True, False, True]
    )
    assert torch.equal(mask[3], expected_pattern)


def test_dilated_self_attention_different_dilation_rates() -> None:
    """Test DilatedSelfAttention with different dilation rates."""
    batch_size, seq_len, d_model = 1, 10, 32
    x = torch.randn(batch_size, seq_len, d_model)

    # Test different dilation rates
    for dilation_rate in [1, 2, 3, 5]:
        attention = DilatedSelfAttention(d_model=d_model, dilation_rate=dilation_rate)
        attention.eval()

        output, weights = attention(x)
        assert output.shape == (batch_size, seq_len, d_model)
        assert weights.shape == (batch_size, 1, seq_len, seq_len)

        mask = create_dilated_mask(seq_len, dilation_rate, x.device)
        sparsity = mask.float().mean().item()

        # Higher dilation rate should generally lead to sparser attention
        # (though this depends on sequence length)
        assert 0 < sparsity <= 1.0


def test_dilated_self_attention_mask_handling() -> None:
    """Test DilatedSelfAttention mask handling."""
    batch_size, seq_len, d_model = 2, 6, 32

    attention = DilatedSelfAttention(d_model=d_model, dilation_rate=2, num_heads=2)
    x = torch.randn(batch_size, seq_len, d_model)

    # Test causal mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    output, attention_weights = attention(x, mask=causal_mask)

    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (batch_size, 2, seq_len, seq_len)

    # Test batch-specific mask
    batch_mask = torch.ones(batch_size, seq_len, seq_len)
    batch_mask[0, :, -1] = 0  # Mask last position for first batch
    output, attention_weights = attention(x, mask=batch_mask)

    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (batch_size, 2, seq_len, seq_len)


def test_dilated_self_attention_temperature_scaling() -> None:
    """Test DilatedSelfAttention temperature scaling effects."""
    batch_size, seq_len, d_model = 1, 4, 16

    x = torch.randn(batch_size, seq_len, d_model)

    # Low temperature (sharper attention)
    attention_low = DilatedSelfAttention(
        d_model=d_model, dilation_rate=2, temperature=0.1
    )
    attention_low.eval()

    # High temperature (smoother attention)
    attention_high = DilatedSelfAttention(
        d_model=d_model, dilation_rate=2, temperature=10.0
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


def test_dilated_self_attention_batch_independence() -> None:
    """Test DilatedSelfAttention batch independence."""
    batch_size, seq_len, d_model = 3, 5, 24

    attention = DilatedSelfAttention(d_model=d_model, dilation_rate=2, num_heads=3)
    attention.eval()

    # Create batch input
    x_batch = torch.randn(batch_size, seq_len, d_model)

    # Process entire batch
    output_batch, weights_batch = attention(x_batch)

    # Process individual samples
    for i in range(batch_size):
        x_single = x_batch[i : i + 1]  # Keep batch dimension
        output_single, weights_single = attention(x_single)

        # Compare results
        assert torch.allclose(output_batch[i : i + 1], output_single, atol=1e-6)
        assert torch.allclose(weights_batch[i : i + 1], weights_single, atol=1e-6)
