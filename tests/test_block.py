"""Function-based test cases for BlockSelfAttention forward method."""

import torch

from attzoo.block import BlockSelfAttention


def test_block_forward_basic_shapes() -> None:
    """Test forward method produces correct output shapes."""
    batch_size, seq_len, d_model = 2, 128, 64
    block_size = 32
    attention = BlockSelfAttention(d_model=d_model, block_size=block_size)
    x = torch.randn(batch_size, seq_len, d_model)

    output, attention_weights = attention(x)

    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (
        batch_size,
        attention.num_heads,
        seq_len,
        seq_len,
    )
    assert isinstance(output, torch.Tensor)
    assert isinstance(attention_weights, torch.Tensor)


def test_block_forward_flexible_input_dimensions() -> None:
    """Test forward method with different input dimensions."""
    batch_size, seq_len = 3, 96
    input_dim, d_model = 32, 64
    block_size = 24

    attention = BlockSelfAttention(
        d_model=d_model, input_dim=input_dim, block_size=block_size
    )
    x = torch.randn(batch_size, seq_len, input_dim)

    output, attention_weights = attention(x)

    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (
        batch_size,
        attention.num_heads,
        seq_len,
        seq_len,
    )


def test_block_forward_different_block_sizes() -> None:
    """Test forward method with different block sizes."""
    batch_size, seq_len, d_model = 2, 128, 64

    for block_size in [16, 32, 64, 128]:
        attention = BlockSelfAttention(d_model=d_model, block_size=block_size)
        x = torch.randn(batch_size, seq_len, d_model)

        output, attention_weights = attention(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (
            batch_size,
            attention.num_heads,
            seq_len,
            seq_len,
        )


def test_block_forward_short_sequences() -> None:
    """Test forward method with sequences shorter than block size."""
    batch_size, d_model = 2, 64
    block_size = 64

    for seq_len in [16, 32, 48]:
        attention = BlockSelfAttention(d_model=d_model, block_size=block_size)
        x = torch.randn(batch_size, seq_len, d_model)

        output, attention_weights = attention(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (
            batch_size,
            attention.num_heads,
            seq_len,
            seq_len,
        )


def test_block_forward_with_padding_mask() -> None:
    """Test forward method with padding mask."""
    batch_size, seq_len, d_model = 2, 128, 64
    block_size = 32
    attention = BlockSelfAttention(d_model=d_model, block_size=block_size)
    attention.eval()

    x = torch.randn(batch_size, seq_len, d_model)
    mask = torch.ones(seq_len, seq_len).bool()
    mask[:, -16:] = False

    output, attention_weights = attention(x, mask=mask)

    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (
        batch_size,
        attention.num_heads,
        seq_len,
        seq_len,
    )


def test_block_forward_different_head_counts() -> None:
    """Test forward method with different numbers of heads."""
    batch_size, seq_len, d_model = 2, 96, 64
    block_size = 24

    for num_heads in [1, 2, 4, 8]:
        attention = BlockSelfAttention(
            d_model=d_model, block_size=block_size, num_heads=num_heads
        )
        x = torch.randn(batch_size, seq_len, d_model)

        output, attention_weights = attention(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)


def test_block_forward_overlapping_blocks() -> None:
    """Test forward method with overlapping blocks."""
    batch_size, seq_len, d_model = 2, 128, 64
    block_size = 32

    attention_overlap = BlockSelfAttention(
        d_model=d_model, block_size=block_size, overlap=True
    )
    x = torch.randn(batch_size, seq_len, d_model)

    output_overlap, weights_overlap = attention_overlap(x)

    assert output_overlap.shape == (batch_size, seq_len, d_model)
    assert weights_overlap.shape == (
        batch_size,
        attention_overlap.num_heads,
        seq_len,
        seq_len,
    )


def test_block_forward_sequence_length_variation() -> None:
    """Test forward method with different sequence lengths."""
    batch_size, d_model = 1, 32
    block_size = 16
    attention = BlockSelfAttention(d_model=d_model, block_size=block_size)

    for seq_len in [8, 16, 32, 48, 64, 96, 128]:
        x = torch.randn(batch_size, seq_len, d_model)
        output, attention_weights = attention(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (
            batch_size,
            attention.num_heads,
            seq_len,
            seq_len,
        )


def test_block_forward_initialization_errors() -> None:
    """Test that proper errors are raised for invalid initialization."""
    d_model = 32

    # Invalid block size
    BlockSelfAttention(d_model=d_model, block_size=0)

    # d_model not divisible by num_heads
    BlockSelfAttention(d_model=33, num_heads=8)
