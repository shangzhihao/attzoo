from typing import Any

import torch
import torch.nn as nn

from attzoo.base import BaseSelfAttention, scaled_dot_product_attention
from attzoo.utils import reshape_from_attention


MASK_DIM_BATCH = 3
MASK_DIM_SEQUENCE = 2


class GroupedSelfAttention(BaseSelfAttention):
    """Grouped Self-Attention implementation (Grouped Query Attention).

    This class implements grouped self-attention where queries are divided into
    groups, but keys and values are shared across all groups. This reduces
    memory usage compared to standard multi-head attention while maintaining
    most of the representation capacity.

    Args:
        d_model: Model dimension (must be divisible by num_query_heads)
        num_query_heads: Number of query heads
        num_kv_heads: Number of key-value heads (must divide num_query_heads evenly)
        input_dim: Input feature dimension (default: None, uses d_model)
        dropout: Dropout probability (default: 0.1)
        bias: Whether to use bias in linear layers (default: False)
        temperature: Temperature scaling for attention scores (default: 1.0)
        rope: Whether to apply Rotary Position Embedding (default: False)
    """

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        input_dim: int | None = None,
        dropout: float = 0.1,
        *,
        bias: bool = False,
        temperature: float = 1.0,
        rope: bool = False,
    ):
        if d_model % num_query_heads != 0:
            message = (
                f"d_model ({d_model}) must be divisible by num_query_heads "
                f"({num_query_heads})"
            )
            raise ValueError(message)

        if num_query_heads % num_kv_heads != 0:
            message = (
                f"num_query_heads ({num_query_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads})"
            )
            raise ValueError(message)

        super().__init__(d_model, input_dim, dropout, bias=bias, rope=rope)

        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.d_head = d_model // num_query_heads
        self.d_kv = (
            d_model // num_query_heads
        )  # Keep same head dimension for compatibility
        self.group_size = num_query_heads // num_kv_heads
        self.temperature = temperature

        # Linear projections for query, key, value, and output
        self.w_q = nn.Linear(self.input_dim, d_model, bias=bias)
        self.w_k = nn.Linear(self.input_dim, num_kv_heads * self.d_kv, bias=bias)
        self.w_v = nn.Linear(self.input_dim, num_kv_heads * self.d_kv, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

        # Initialize weights
        self._init_weights()

    def _reshape_queries_for_attention(self, q: torch.Tensor) -> torch.Tensor:
        """Reshape query tensor for grouped attention.

        Args:
            q: Query tensor [batch_size, seq_len, d_model]

        Returns:
            Reshaped tensor [batch_size, num_query_heads, seq_len, d_head]
        """
        batch_size, seq_len, _d_model = q.shape
        # Reshape to [batch_size, seq_len, num_query_heads, d_head]
        q = q.view(batch_size, seq_len, self.num_query_heads, self.d_head)
        # Transpose to [batch_size, num_query_heads, seq_len, d_head]
        return q.transpose(1, 2)

    def _reshape_kv_for_attention(self, kv: torch.Tensor) -> torch.Tensor:
        """Reshape key/value tensor for grouped attention.

        Args:
            kv: Key or value tensor [batch_size, seq_len, num_kv_heads * d_kv]

        Returns:
            Reshaped tensor [batch_size, num_kv_heads, seq_len, d_kv]
        """
        batch_size, seq_len, _ = kv.shape
        # Reshape to [batch_size, seq_len, num_kv_heads, d_kv]
        kv = kv.view(batch_size, seq_len, self.num_kv_heads, self.d_kv)
        # Transpose to [batch_size, num_kv_heads, seq_len, d_kv]
        return kv.transpose(1, 2)

    def _expand_kv_for_groups(self, kv: torch.Tensor) -> torch.Tensor:
        """Expand key/value tensor to match query groups.

        Args:
            kv: Key or value tensor [batch_size, num_kv_heads, seq_len, d_kv]

        Returns:
            Expanded tensor [batch_size, num_query_heads, seq_len, d_kv]
        """
        # Repeat each kv head for group_size query heads
        # Shape: [batch_size, num_kv_heads, seq_len, d_kv] -> [batch_size, num_query_heads, seq_len, d_kv]
        return kv.repeat_interleave(self.group_size, dim=1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of grouped self-attention.

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask [batch_size, seq_len, seq_len] or broadcastable
            **kwargs: Additional arguments

        Returns:
            Tuple of (output, attention_weights)
            - output: [batch_size, seq_len, d_model]
            - attention_weights: [batch_size, num_query_heads, seq_len, seq_len]
        """
        batch_size, _seq_len, _ = x.shape

        # Apply linear projections
        q = self.w_q(x)  # [batch_size, seq_len, d_model]
        k = self.w_k(x)  # [batch_size, seq_len, num_kv_heads * d_kv]
        v = self.w_v(x)  # [batch_size, seq_len, num_kv_heads * d_kv]

        # Reshape for grouped attention
        q = self._reshape_queries_for_attention(
            q
        )  # [batch_size, num_query_heads, seq_len, d_head]
        k = self._reshape_kv_for_attention(
            k
        )  # [batch_size, num_kv_heads, seq_len, d_kv]
        v = self._reshape_kv_for_attention(
            v
        )  # [batch_size, num_kv_heads, seq_len, d_kv]

        # Apply RoPE if enabled (before expanding k and v)
        if self.rope:
            q, k = self.apply_rope(q, k)

        # Expand k and v to match query groups
        k = self._expand_kv_for_groups(
            k
        )  # [batch_size, num_query_heads, seq_len, d_kv]
        v = self._expand_kv_for_groups(
            v
        )  # [batch_size, num_query_heads, seq_len, d_kv]

        # Expand mask for multiple query heads if provided
        if mask is not None:
            mask_rank = mask.dim()
            if mask_rank == MASK_DIM_BATCH:
                mask = mask.unsqueeze(1).expand(-1, self.num_query_heads, -1, -1)
            elif mask_rank == MASK_DIM_SEQUENCE:
                mask = (
                    mask.unsqueeze(0)
                    .unsqueeze(0)
                    .expand(batch_size, self.num_query_heads, -1, -1)
                )

        # Compute scaled dot-product attention
        attention_output, attention_weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout, temperature=self.temperature
        )

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
                "num_query_heads": self.num_query_heads,
                "num_kv_heads": self.num_kv_heads,
                "d_head": self.d_head,
                "d_kv": self.d_kv,
                "group_size": self.group_size,
                "temperature": self.temperature,
            }
        )
        return config

    def extra_repr(self) -> str:
        """Extra representation for debugging."""
        base_repr = super().extra_repr()
        return (
            f"{base_repr}, num_query_heads={self.num_query_heads}, "
            f"num_kv_heads={self.num_kv_heads}, d_head={self.d_head}, "
            f"group_size={self.group_size}, temperature={self.temperature}"
        )
