from typing import Any

import torch
import torch.nn as nn

from attzoo.base import BaseSelfAttention, scaled_dot_product_attention
from attzoo.masks import create_local_mask
from attzoo.utils import reshape_for_attention, reshape_from_attention


GLOBAL_ATTENTION_RANK_SEQUENCE = 1
GLOBAL_ATTENTION_RANK_BATCH = 2
MASK_DIM_SEQUENCE = 2
MASK_DIM_BATCH = 3
MASK_DIM_BATCH_HEAD = 4


class LongformerSelfAttention(BaseSelfAttention):
    """Longformer-style sliding window self-attention with optional global tokens.

    Implements the Longformer attention pattern:
    - Sliding window local attention of size ``window_size`` around each token
    - Optional global tokens which can attend to all tokens and be attended by all

    Args:
        d_model: Model dimension (must be divisible by num_heads)
        num_heads: Number of attention heads
        input_dim: Input feature dimension (default: None, uses d_model)
        window_size: Sliding window size (default: 512)
        dropout: Dropout probability (default: 0.1)
        bias: Whether to use bias in linear layers (default: False)
        temperature: Temperature scaling for attention scores (default: 1.0)
        rope: Whether to apply Rotary Position Embedding (default: False)

    Forward inputs:
        x: Tensor of shape [batch, seq, input_dim]
        mask: Optional attention mask. Supports:
            - bool: [seq, seq], [batch, seq, seq], or [batch, num_heads, seq, seq]
            - additive: same shapes, where masked positions are large negative
        global_attention: Optional bool mask indicating global tokens.
            Shapes: [seq] or [batch, seq]. True means global.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        input_dim: int | None = None,
        window_size: int = 512,
        dropout: float = 0.1,
        *,
        bias: bool = False,
        temperature: float = 1.0,
        rope: bool = False,
    ):
        if window_size <= 0:
            message = f"window_size must be positive, got {window_size}"
            raise ValueError(message)
        if d_model % num_heads != 0:
            message = (
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
            raise ValueError(message)

        super().__init__(d_model, input_dim, dropout, bias=bias, rope=rope)

        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.window_size = window_size
        self.temperature = temperature

        # Projections
        self.w_q = nn.Linear(self.input_dim, d_model, bias=bias)
        self.w_k = nn.Linear(self.input_dim, d_model, bias=bias)
        self.w_v = nn.Linear(self.input_dim, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

        self._init_weights()

    def _build_longformer_mask(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        base_mask: torch.Tensor | None,
        global_attention: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Create a combined Longformer attention mask expanded to heads."""
        long_mask = self._create_local_mask(batch_size, seq_len, device)
        long_mask = self._apply_global_attention(
            long_mask, global_attention, batch_size, seq_len
        )
        return self._merge_with_base_mask(
            long_mask, base_mask, batch_size, seq_len, device
        )

    def _create_local_mask(
        self, batch_size: int, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        local = create_local_mask(seq_len, self.window_size, device)
        return (
            local.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
        )

    def _apply_global_attention(
        self,
        long_mask: torch.Tensor,
        global_attention: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        if global_attention is None:
            return long_mask

        normalized = self._normalize_global_attention(
            global_attention, batch_size, seq_len
        )
        g_bool = normalized.bool()
        g_cols = g_bool.view(batch_size, 1, 1, seq_len).expand(
            -1, self.num_heads, seq_len, -1
        )
        g_rows = g_bool.view(batch_size, 1, seq_len, 1).expand(
            -1, self.num_heads, -1, seq_len
        )
        return long_mask | g_cols | g_rows

    def _normalize_global_attention(
        self,
        global_attention: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        if global_attention.dim() == GLOBAL_ATTENTION_RANK_SEQUENCE:
            return global_attention.unsqueeze(0).expand(batch_size, -1)
        if global_attention.dim() == GLOBAL_ATTENTION_RANK_BATCH:
            return global_attention
        message = "global_attention must be [seq] or [batch, seq] boolean mask"
        raise ValueError(message)

    def _merge_with_base_mask(
        self,
        long_mask: torch.Tensor,
        base_mask: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if base_mask is None:
            return long_mask
        expanded = self._expand_base_mask(base_mask, batch_size, seq_len)
        if expanded.dtype == torch.bool:
            return long_mask & expanded
        additive_long = torch.where(
            long_mask,
            torch.tensor(0.0, device=device, dtype=expanded.dtype),
            torch.tensor(float("-inf"), device=device, dtype=expanded.dtype),
        )
        return additive_long + expanded

    def _expand_base_mask(
        self, base_mask: torch.Tensor, batch_size: int, seq_len: int
    ) -> torch.Tensor:
        if base_mask.dim() == MASK_DIM_SEQUENCE:
            return (
                base_mask.unsqueeze(0)
                .unsqueeze(0)
                .expand(batch_size, self.num_heads, -1, -1)
            )
        if base_mask.dim() == MASK_DIM_BATCH:
            return base_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        if base_mask.dim() == MASK_DIM_BATCH_HEAD:
            return base_mask
        message = "mask must be 2D/3D/4D tensor"
        raise ValueError(message)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        global_attention: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        # Linear projections
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # Reshape to heads
        q = reshape_for_attention(q, self.num_heads, self.d_head)
        k = reshape_for_attention(k, self.num_heads, self.d_head)
        v = reshape_for_attention(v, self.num_heads, self.d_head)

        # RoPE if enabled
        if self.rope:
            q, k = self.apply_rope(q, k)

        # Build Longformer mask and move/cast to match q/k dtype/device if additive
        combined_mask = self._build_longformer_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            device=x.device,
            base_mask=mask,
            global_attention=global_attention,
        )

        # Compute attention
        attn_out, attn_weights = scaled_dot_product_attention(
            q,
            k,
            v,
            mask=combined_mask,
            dropout=self.dropout,
            temperature=self.temperature,
        )

        # Back to [batch, seq, d_model]
        attn_out = reshape_from_attention(attn_out, self.d_model)
        output = self.w_o(attn_out)

        # Store averaged attention weights for convenience
        self.attention_weights = attn_weights.mean(dim=1).detach()

        return output, attn_weights

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "d_head": self.d_head,
                "window_size": self.window_size,
                "temperature": self.temperature,
            }
        )
        return config

    def extra_repr(self) -> str:
        base = super().extra_repr()
        return f"{base}, num_heads={self.num_heads}, d_head={self.d_head}, window_size={self.window_size}, temperature={self.temperature}"
