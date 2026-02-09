import pytest
import torch

from attentions.group import GroupedSelfAttention


def test_grouped_self_attention_initialization() -> None:
    """Test GroupedSelfAttention initialization with various configurations."""
    # Test basic initialization
    attention = GroupedSelfAttention(d_model=64, num_query_heads=8, num_kv_heads=2)
    assert attention.d_model == 64
    assert attention.num_query_heads == 8
    assert attention.num_kv_heads == 2
    assert attention.d_head == 8  # 64 / 8
    assert attention.group_size == 4  # 8 / 2
    assert attention.input_dim == 64  # defaults to d_model
    assert attention.dropout_prob == 0.1  # default
    assert attention.temperature == 1.0  # default

    # Test with custom input_dim
    attention = GroupedSelfAttention(
        d_model=128, num_query_heads=16, num_kv_heads=4, input_dim=256
    )
    assert attention.d_model == 128
    assert attention.input_dim == 256
    assert attention.num_query_heads == 16
    assert attention.num_kv_heads == 4
    assert attention.d_head == 8  # 128 / 16
    assert attention.group_size == 4  # 16 / 4

    # Test with custom parameters
    attention = GroupedSelfAttention(
        d_model=96,
        num_query_heads=12,
        num_kv_heads=3,
        dropout=0.2,
        bias=False,
        temperature=0.8,
    )
    assert attention.dropout_prob == 0.2
    assert not attention.bias
    assert attention.temperature == 0.8


def test_grouped_self_attention_initialization_errors() -> None:
    """Test GroupedSelfAttention initialization error cases."""
    # Test d_model not divisible by num_query_heads
    with pytest.raises(
        ValueError,
        match=r"d_model \(\d+\) must be divisible by num_query_heads \(\d+\)",
    ):
        GroupedSelfAttention(d_model=64, num_query_heads=7, num_kv_heads=1)

    # Test num_query_heads not divisible by num_kv_heads
    with pytest.raises(
        ValueError,
        match=r"num_query_heads \(\d+\) must be divisible by num_kv_heads \(\d+\)",
    ):
        GroupedSelfAttention(d_model=64, num_query_heads=8, num_kv_heads=3)


def test_grouped_self_attention_forward_shapes() -> None:
    """Test GroupedSelfAttention forward pass output shapes."""
    batch_size, seq_len, d_model = 2, 10, 64
    num_query_heads, num_kv_heads = 8, 2

    attention = GroupedSelfAttention(
        d_model=d_model, num_query_heads=num_query_heads, num_kv_heads=num_kv_heads
    )
    x = torch.randn(batch_size, seq_len, d_model)

    output, attention_weights = attention(x)

    # Check output shapes
    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (batch_size, num_query_heads, seq_len, seq_len)

    # Check stored attention weights (averaged across heads)
    stored_weights = attention.get_attention_weights()
    assert stored_weights.shape == (batch_size, seq_len, seq_len)


def test_grouped_self_attention_forward_flexible_input() -> None:
    """Test GroupedSelfAttention with flexible input dimensions."""
    batch_size, seq_len, input_dim, d_model = 2, 8, 128, 64
    num_query_heads, num_kv_heads = 8, 4

    attention = GroupedSelfAttention(
        d_model=d_model,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        input_dim=input_dim,
    )
    x = torch.randn(batch_size, seq_len, input_dim)

    output, attention_weights = attention(x)

    # Output should have d_model dimensions
    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (batch_size, num_query_heads, seq_len, seq_len)


def test_grouped_self_attention_mask_handling() -> None:
    """Test GroupedSelfAttention mask handling."""
    batch_size, seq_len, d_model = 2, 6, 32
    num_query_heads, num_kv_heads = 4, 2

    attention = GroupedSelfAttention(
        d_model=d_model, num_query_heads=num_query_heads, num_kv_heads=num_kv_heads
    )
    x = torch.randn(batch_size, seq_len, d_model)

    # Test causal mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    output, attention_weights = attention(x, mask=causal_mask)

    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (batch_size, num_query_heads, seq_len, seq_len)

    # Test batch-specific mask
    batch_mask = torch.ones(batch_size, seq_len, seq_len)
    batch_mask[0, :, -1] = 0  # Mask last position for first batch
    output, attention_weights = attention(x, mask=batch_mask)

    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (batch_size, num_query_heads, seq_len, seq_len)


def test_grouped_self_attention_deterministic() -> None:
    """Test GroupedSelfAttention produces deterministic results."""
    torch.manual_seed(42)

    batch_size, seq_len, d_model = 1, 5, 32
    num_query_heads, num_kv_heads = 4, 1

    attention = GroupedSelfAttention(
        d_model=d_model, num_query_heads=num_query_heads, num_kv_heads=num_kv_heads
    )
    attention.eval()  # Disable dropout for deterministic behavior

    x = torch.randn(batch_size, seq_len, d_model)

    # First forward pass
    output1, weights1 = attention(x)

    # Second forward pass with same input
    output2, weights2 = attention(x)

    # Should be identical
    assert torch.allclose(output1, output2, atol=1e-6)
    assert torch.allclose(weights1, weights2, atol=1e-6)


def test_grouped_self_attention_temperature_scaling() -> None:
    """Test GroupedSelfAttention temperature scaling effects."""
    batch_size, seq_len, d_model = 1, 3, 16
    num_query_heads, num_kv_heads = 4, 2

    x = torch.randn(batch_size, seq_len, d_model)

    # Low temperature (sharper attention)
    attention_low = GroupedSelfAttention(
        d_model=d_model,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        temperature=0.1,
    )
    attention_low.eval()

    # High temperature (smoother attention)
    attention_high = GroupedSelfAttention(
        d_model=d_model,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        temperature=10.0,
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
