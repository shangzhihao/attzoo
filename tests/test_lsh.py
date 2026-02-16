"""Function-based test cases for LSHSelfAttention."""

import pytest
import torch

from attzoo.lsh import LSHSelfAttention


def test_lsh_forward_basic_shapes() -> None:
    """LSH attention should return outputs and weights with expected shapes."""
    torch.manual_seed(0)

    batch_size, seq_len, d_model = 2, 24, 32
    num_heads, bucket_size, num_hashes = 4, 8, 3
    attention = LSHSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        bucket_size=bucket_size,
        num_hashes=num_hashes,
        dropout=0.0,
    )

    x = torch.randn(batch_size, seq_len, d_model)

    output, attention_weights = attention(x)
    assert attention_weights is not None

    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)
    stored_weights = attention.attention_weights
    assert stored_weights is not None
    assert stored_weights.shape == (batch_size, seq_len, seq_len)
    assert torch.allclose(stored_weights, attention_weights.mean(dim=1))

    row_sums = attention_weights.sum(dim=-1)
    ones = torch.ones_like(row_sums)
    assert torch.allclose(row_sums, ones, atol=1e-5)


def test_lsh_forward_with_boolean_mask() -> None:
    """Boolean masks should zero-out prohibited attention positions."""
    torch.manual_seed(1)

    batch_size, seq_len, d_model = 1, 16, 32
    num_heads, bucket_size = 2, 4
    attention = LSHSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        bucket_size=bucket_size,
        dropout=0.0,
    )

    x = torch.randn(batch_size, seq_len, d_model)
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
    mask[:, -3:] = False

    _, attention_weights = attention(x, mask=mask)
    assert attention_weights is not None

    expanded_mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
    masked_values = attention_weights[~expanded_mask]
    assert torch.all(masked_values == 0)


def test_lsh_initialization_errors() -> None:
    """Constructor should validate divisibility, bucket size, and hash count."""
    with pytest.raises(
        ValueError, match=r"d_model \(33\) must be divisible by num_heads \(8\)"
    ):
        LSHSelfAttention(d_model=33, num_heads=8)

    with pytest.raises(ValueError, match="bucket_size must be positive"):
        LSHSelfAttention(d_model=32, num_heads=4, bucket_size=0)

    with pytest.raises(ValueError, match="num_hashes must be positive"):
        LSHSelfAttention(d_model=32, num_heads=4, num_hashes=0)
