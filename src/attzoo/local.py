from typing import Any

import torch
import torch.nn as nn

from attzoo.base import BaseSelfAttention, scaled_dot_product_attention
from attzoo.masks import create_local_mask
from attzoo.utils import reshape_for_attention, reshape_from_attention


MASK_DIM_SEQUENCE = 2
MASK_DIM_BATCH = 3
MASK_DIM_BATCH_HEAD = 4


class LocalSelfAttention(BaseSelfAttention):
    """Local Multi-Head Self-Attention implementation.

    This class implements local self-attention with multiple heads where each position
    only attends to a local window of positions around it. This reduces computational
    complexity from O(n²) to O(n*w) where w is the window size, while maintaining
    the representational power of multi-head attention.

    Args:
        d_model: Model dimension (must be divisible by num_heads)
        input_dim: Input feature dimension (default: None, uses d_model)
        window_size: Size of the local attention window (default: 128)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.1)
        bias: Whether to use bias in linear layers (default: False)
        temperature: Temperature scaling for attention scores (default: 1.0)
        rope: Whether to apply Rotary Position Embedding (default: False)
    """

    def __init__(
        self,
        d_model: int,
        input_dim: int | None = None,
        window_size: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        *,
        bias: bool = False,
        temperature: float = 1.0,
        rope: bool = False,
    ):
        super().__init__(d_model, input_dim, dropout, bias=bias, rope=rope)

        if window_size <= 0:
            message = f"window_size must be positive, got {window_size}"
            raise ValueError(message)

        if d_model % num_heads != 0:
            message = (
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
            raise ValueError(message)

        self.window_size = window_size
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.temperature = temperature

        # Linear projections for query, key, value, and output
        self.w_q = nn.Linear(self.input_dim, d_model, bias=bias)
        self.w_k = nn.Linear(self.input_dim, d_model, bias=bias)
        self.w_v = nn.Linear(self.input_dim, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

        # Initialize weights
        self._init_weights()

    def _apply_local_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply local multi-head attention with windowing.

        Args:
            query: Query tensor [batch_size, num_heads, seq_len, d_head]
            key: Key tensor [batch_size, num_heads, seq_len, d_head]
            value: Value tensor [batch_size, num_heads, seq_len, d_head]
            mask: Optional additional mask

        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, num_heads, seq_len, _d_head = query.shape

        # Create local attention mask
        local_mask = create_local_mask(seq_len, self.window_size, query.device)

        # Expand local mask for multiple heads
        local_mask_expanded = (
            local_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
        )

        # Combine with additional mask if provided
        if mask is not None:
            mask_rank = mask.dim()
            if mask_rank == MASK_DIM_SEQUENCE:
                mask = (
                    mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
                )
            elif mask_rank == MASK_DIM_BATCH:
                mask = mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
            elif mask_rank == MASK_DIM_BATCH_HEAD:
                pass  # Already in correct format

            # Apply both local mask and provided mask
            combined_mask = local_mask_expanded & mask.bool()
        else:
            combined_mask = local_mask_expanded

        # Apply scaled dot-product attention with the combined mask
        output, attention_weights = scaled_dot_product_attention(
            query,
            key,
            value,
            mask=combined_mask,
            dropout=self.dropout,
            temperature=self.temperature,
        )

        return output, attention_weights

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of local multi-head self-attention.

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask to combine with local mask
            **kwargs: Additional arguments

        Returns:
            Tuple of (output, attention_weights)
            - output: [batch_size, seq_len, d_model]
            - attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        # Apply linear projections
        q = self.w_q(x)  # [batch_size, seq_len, d_model]
        k = self.w_k(x)  # [batch_size, seq_len, d_model]
        v = self.w_v(x)  # [batch_size, seq_len, d_model]

        # Reshape for multi-head attention
        q = reshape_for_attention(q, self.num_heads, self.d_head)
        k = reshape_for_attention(k, self.num_heads, self.d_head)
        v = reshape_for_attention(v, self.num_heads, self.d_head)

        # Apply RoPE if enabled
        if self.rope:
            q, k = self.apply_rope(q, k)

        # Apply local attention with multiple heads
        attention_output, attention_weights = self._apply_local_attention(q, k, v, mask)

        # Reshape back to original format
        attention_output = reshape_from_attention(attention_output, self.d_model)

        # Apply output projection
        output = self.w_o(attention_output)

        # Average attention weights across heads for monitoring
        self.attention_weights = attention_weights.mean(dim=1).detach()

        return output, attention_weights

    def get_config(self) -> dict[str, Any]:
        """Get configuration dictionary."""
        config = super().get_config()
        config.update(
            {
                "window_size": self.window_size,
                "num_heads": self.num_heads,
                "d_head": self.d_head,
                "temperature": self.temperature,
            }
        )
        return config

    def extra_repr(self) -> str:
        """Extra representation for debugging."""
        base_repr = super().extra_repr()
        return (
            f"{base_repr}, window_size={self.window_size}, num_heads={self.num_heads}, "
            f"d_head={self.d_head}, temperature={self.temperature}"
        )
