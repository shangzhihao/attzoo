from typing import Any

import torch

from attentions.base import BaseSelfAttention, scaled_dot_product_attention
from attentions.utils import reshape_for_attention, reshape_from_attention


MASK_RANK_BATCH = 3
MASK_RANK_PAIRWISE = 2


class MultiHeadSelfAttention(BaseSelfAttention):
    """Multi-Head Self-Attention implementation.

    This class implements the multi-head self-attention mechanism where the input
    is linearly projected into multiple heads, attention is computed in parallel
    for each head, and the results are concatenated and projected back.

    Args:
        d_model: Model dimension (must be divisible by num_heads)
        num_heads: Number of attention heads
        input_dim: Input feature dimension (default: None, uses d_model)
        dropout: Dropout probability (default: 0.1)
        bias: Whether to use bias in linear layers (default: False)
        temperature: Temperature scaling for attention scores (default: 1.0)
        rope: Whether to apply Rotary Position Embedding (default: False)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        input_dim: int | None = None,
        dropout: float = 0.1,
        *,
        bias: bool = False,
        temperature: float = 1.0,
        rope: bool = False,
    ):
        if d_model % num_heads != 0:
            message = (
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
            raise ValueError(message)

        super().__init__(d_model, input_dim, dropout, bias=bias, rope=rope)

        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.temperature = temperature

        # Create standard linear projection layers
        # Using single linear layers and reshaping for efficiency
        self.w_q, self.w_k, self.w_v, self.w_o = self._create_projection_layers(
            bias=bias
        )

        # Initialize weights
        self._init_weights()

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of multi-head self-attention.

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask [batch_size, seq_len, seq_len] or broadcastable
            **kwargs: Additional arguments

        Returns:
            Tuple of (output, attention_weights)
            - output: [batch_size, seq_len, d_model]
            - attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, _seq_len, _d_model = x.shape

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

        # Expand mask for multiple heads if provided
        if mask is not None:
            mask_rank = mask.dim()
            if mask_rank == MASK_RANK_BATCH:
                mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            elif mask_rank == MASK_RANK_PAIRWISE:
                mask = (
                    mask.unsqueeze(0)
                    .unsqueeze(0)
                    .expand(batch_size, self.num_heads, -1, -1)
                )

        # Compute scaled dot-product attention for each head
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
            f"{base_repr}, num_heads={self.num_heads}, d_head={self.d_head}, "
            f"temperature={self.temperature}"
        )
