"""Tensor reshaping utilities for attention mechanisms.

This module provides common tensor reshaping operations used across different
attention mechanism implementations to reduce code duplication.
"""

import torch


def reshape_for_attention(x: torch.Tensor, num_heads: int, d_head: int) -> torch.Tensor:
    """Reshape tensor for multi-head attention.

    Transforms input tensor from [batch_size, seq_len, d_model] to
    [batch_size, num_heads, seq_len, d_head] format for multi-head attention computation.

    Args:
        x: Input tensor [batch_size, seq_len, d_model]
        num_heads: Number of attention heads
        d_head: Dimension per head (d_model // num_heads)

    Returns:
        Reshaped tensor [batch_size, num_heads, seq_len, d_head]

    Raises:
        ValueError: If d_model is not equal to num_heads * d_head
    """
    batch_size, seq_len, d_model = x.shape

    # Validate dimensions
    expected_d_model = num_heads * d_head
    if d_model != expected_d_model:
        message = (
            f"d_model ({d_model}) must equal num_heads * d_head "
            f"({num_heads} * {d_head} = {expected_d_model})"
        )
        raise ValueError(message)

    # Reshape to [batch_size, seq_len, num_heads, d_head]
    x = x.view(batch_size, seq_len, num_heads, d_head)
    # Transpose to [batch_size, num_heads, seq_len, d_head]
    return x.transpose(1, 2)


def reshape_from_attention(x: torch.Tensor, d_model: int) -> torch.Tensor:
    """Reshape tensor back from multi-head attention format.

    Transforms tensor from [batch_size, num_heads, seq_len, d_head] to
    [batch_size, seq_len, d_model] format after multi-head attention computation.

    Args:
        x: Input tensor [batch_size, num_heads, seq_len, d_head]
        d_model: Model dimension (num_heads * d_head)

    Returns:
        Reshaped tensor [batch_size, seq_len, d_model]

    Raises:
        ValueError: If computed d_model doesn't match expected d_model
    """
    batch_size, num_heads, seq_len, d_head = x.shape

    # Validate dimensions
    computed_d_model = num_heads * d_head
    if d_model != computed_d_model:
        message = (
            f"d_model ({d_model}) must equal num_heads * d_head "
            f"({num_heads} * {d_head} = {computed_d_model})"
        )
        raise ValueError(message)

    # Transpose to [batch_size, seq_len, num_heads, d_head]
    x = x.transpose(1, 2)
    # Reshape to [batch_size, seq_len, d_model]
    return x.contiguous().view(batch_size, seq_len, d_model)
