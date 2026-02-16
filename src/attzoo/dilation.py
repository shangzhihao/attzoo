from typing import Any

import torch
import torch.nn as nn

from attzoo.base import BaseSelfAttention, scaled_dot_product_attention
from attzoo.masks import create_dilated_mask
from attzoo.utils import reshape_for_attention, reshape_from_attention


MASK_DIM_SEQUENCE = 2
MASK_DIM_BATCH = 3


class DilatedSelfAttention(BaseSelfAttention):
    """Dilated Self-Attention implementation.

    This class implements dilated self-attention where attention is computed
    with a dilation pattern, similar to dilated convolutions. This allows
    the model to attend to positions at regular intervals, which can be
    useful for capturing long-range dependencies efficiently while reducing
    computational complexity.

    Args:
        d_model: Model dimension for attention computation
        dilation_rate: Dilation rate for attention pattern (default: 2)
        num_heads: Number of attention heads (default: 1)
        input_dim: Input feature dimension (default: None, uses d_model)
        dropout: Dropout probability (default: 0.1)
        bias: Whether to use bias in linear layers (default: False)
        temperature: Temperature scaling for attention scores (default: 1.0)
        rope: Whether to apply Rotary Position Embedding (default: False)
    """

    def __init__(
        self,
        d_model: int,
        dilation_rate: int = 2,
        num_heads: int = 1,
        input_dim: int | None = None,
        dropout: float = 0.1,
        *,
        bias: bool = False,
        temperature: float = 1.0,
        rope: bool = False,
    ):
        if dilation_rate < 1:
            message = f"dilation_rate must be >= 1, got {dilation_rate}"
            raise ValueError(message)

        if d_model % num_heads != 0:
            message = (
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
            raise ValueError(message)

        super().__init__(d_model, input_dim, dropout, bias=bias, rope=rope)

        self.dilation_rate = dilation_rate
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

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of dilated self-attention.

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask [batch_size, seq_len, seq_len] or broadcastable
            **kwargs: Additional arguments

        Returns:
            Tuple of (output, attention_weights)
            - output: [batch_size, seq_len, d_model]
            - attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        q, k, v = self._project_inputs(x)
        q, k, v = self._reshape_for_heads(q, k, v)
        q, k = self._maybe_apply_rope(q, k)

        dilated_mask = create_dilated_mask(seq_len, self.dilation_rate, x.device)
        combined_mask = self._prepare_combined_mask(
            dilated_mask, mask, batch_size, seq_len
        )

        attention_output, attention_weights = self._compute_dilated_attention(
            q, k, v, combined_mask
        )
        attention_output = self._restore_sequence(attention_output)
        output = self.w_o(attention_output)
        self.attention_weights = self._summarize_attention(attention_weights)

        return output, attention_weights

    def _project_inputs(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.w_q(x), self.w_k(x), self.w_v(x)

    def _reshape_for_heads(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.num_heads > 1:
            return (
                reshape_for_attention(q, self.num_heads, self.d_head),
                reshape_for_attention(k, self.num_heads, self.d_head),
                reshape_for_attention(v, self.num_heads, self.d_head),
            )
        return q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)

    def _maybe_apply_rope(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.rope:
            return q, k
        return self.apply_rope(q, k)

    def _prepare_combined_mask(
        self,
        dilated_mask: torch.Tensor,
        user_mask: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        base_mask = self._expand_dilated_mask(dilated_mask, batch_size)
        if user_mask is None:
            return base_mask

        expanded_user_mask = self._expand_user_mask(user_mask, batch_size, seq_len)
        if expanded_user_mask.dtype == torch.bool:
            return base_mask & expanded_user_mask

        additive_base = torch.where(
            dilated_mask,
            torch.tensor(0.0, device=base_mask.device, dtype=expanded_user_mask.dtype),
            torch.tensor(
                float("-inf"), device=base_mask.device, dtype=expanded_user_mask.dtype
            ),
        )
        additive_base = additive_base.unsqueeze(0).unsqueeze(0).expand_as(base_mask)
        return additive_base + expanded_user_mask

    def _expand_dilated_mask(
        self, dilated_mask: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        return (
            dilated_mask.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, self.num_heads, -1, -1)
        )

    def _expand_user_mask(
        self,
        mask: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        mask_rank = mask.dim()
        if mask_rank == MASK_DIM_SEQUENCE:
            return (
                mask.unsqueeze(0)
                .unsqueeze(0)
                .expand(batch_size, self.num_heads, -1, -1)
            )
        if mask_rank == MASK_DIM_BATCH:
            return mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        message = f"Unsupported mask dimensions: {mask_rank}"
        raise ValueError(message)

    def _compute_dilated_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        combined_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return scaled_dot_product_attention(
            q,
            k,
            v,
            mask=combined_mask,
            dropout=self.dropout,
            temperature=self.temperature,
        )

    def _restore_sequence(self, attention_output: torch.Tensor) -> torch.Tensor:
        if self.num_heads > 1:
            return reshape_from_attention(attention_output, self.d_model)
        return attention_output.squeeze(1)

    def _summarize_attention(self, attention_weights: torch.Tensor) -> torch.Tensor:
        if self.num_heads > 1:
            return attention_weights.mean(dim=1).detach()
        return attention_weights.squeeze(1).detach()

    def get_config(self) -> dict[str, Any]:
        """Get configuration dictionary."""
        config = super().get_config()
        config.update(
            {
                "dilation_rate": self.dilation_rate,
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
            f"{base_repr}, dilation_rate={self.dilation_rate}, "
            f"num_heads={self.num_heads}, d_head={self.d_head}, "
            f"temperature={self.temperature}"
        )
