import math
from collections.abc import Iterable
from typing import Any, cast

import torch
import torch.nn as nn

from attzoo.base import BaseSelfAttention, scaled_dot_product_attention
from attzoo.utils import reshape_for_attention, reshape_from_attention


MASK_RANK_SQUARE = 2
MASK_RANK_BATCH = 3


class AlibiSelfAttention(BaseSelfAttention):
    """ALiBi (Attention with Linear Biases) Self-Attention implementation.

    ALiBi replaces positional embeddings with position-dependent biases that are
    added directly to the attention scores. The biases scale linearly with distance,
    enabling better extrapolation to sequences longer than those seen during training.

    The bias for position (i, j) is computed as: -m * |i - j|
    where m is a head-specific slope derived from: m = 1 / (2^(8/n))^head_idx

    Args:
        d_model: Model dimension (must be divisible by num_heads)
        num_heads: Number of attention heads
        input_dim: Input feature dimension (default: None, uses d_model)
        dropout: Dropout probability (default: 0.1)
        bias: Whether to use bias in linear layers (default: False)
        temperature: Temperature scaling for attention scores (default: 1.0)
        max_seq_len: Maximum sequence length for pre-computed biases (default: 2048)
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
        max_seq_len: int = 2048,
    ):
        if d_model % num_heads != 0:
            message = (
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
            raise ValueError(message)

        super().__init__(d_model, input_dim, dropout, bias=bias)

        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.temperature = temperature
        self.max_seq_len = max_seq_len

        # Linear projections for query, key, value, and output
        self.w_q = nn.Linear(self.input_dim, d_model, bias=bias)
        self.w_k = nn.Linear(self.input_dim, d_model, bias=bias)
        self.w_v = nn.Linear(self.input_dim, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

        # Pre-compute ALiBi slopes for each head
        self.register_buffer("slopes", self._get_alibi_slopes(num_heads))

        # Pre-compute bias matrix for efficiency
        self.register_buffer("alibi_bias", self._create_alibi_bias(max_seq_len))

        # Initialize weights
        self._init_weights()

    def _get_alibi_slopes(self, num_heads: int) -> torch.Tensor:
        """Compute ALiBi slopes for each attention head.

        Args:
            num_heads: Number of attention heads

        Returns:
            Tensor of slopes [num_heads] where each slope is 1/(2^(8/n))^head_idx
        """
        slopes = self._generate_slopes(num_heads)
        self._validate_slopes(slopes, num_heads)
        return torch.tensor(slopes, dtype=torch.float32)

    @staticmethod
    def _generate_slopes(num_heads: int) -> list[float]:
        if num_heads <= 0:
            return []
        if num_heads == 1:
            return [1.0]
        if math.log2(num_heads).is_integer():
            return AlibiSelfAttention._power_of_two_slopes(num_heads)

        base = 2 ** math.floor(math.log2(num_heads))
        slopes = AlibiSelfAttention._power_of_two_slopes(base)
        remaining = num_heads - base
        if remaining == 0:
            return slopes

        extended = AlibiSelfAttention._power_of_two_slopes(base * 2)
        return slopes + extended[1::2][:remaining]

    @staticmethod
    def _power_of_two_slopes(num_heads: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(num_heads) - 3)))
        ratio = start
        return [start * (ratio**index) for index in range(num_heads)]

    @staticmethod
    def _validate_slopes(slopes: Iterable[float], num_heads: int) -> None:
        slopes_list = list(slopes)
        if len(slopes_list) != num_heads:
            message = f"Expected {num_heads} slopes, got {len(slopes_list)}"
            raise ValueError(message)
        if any(slope <= 0 for slope in slopes_list):
            message = "All ALiBi slopes must be positive"
            raise ValueError(message)

    def _create_alibi_bias(self, max_seq_len: int) -> torch.Tensor:
        """Pre-compute ALiBi bias matrix for efficiency.

        Args:
            max_seq_len: Maximum sequence length

        Returns:
            Bias tensor [num_heads, max_seq_len, max_seq_len]
        """
        # Create position distance matrix
        positions = torch.arange(max_seq_len).unsqueeze(0)  # [1, max_seq_len]
        distances = torch.abs(positions.T - positions)  # [max_seq_len, max_seq_len]

        # Apply slopes to distances: bias = -slope * distance
        # self.slopes is already a tensor, no need for torch.as_tensor
        slopes_tensor = cast(torch.Tensor, self.slopes)
        bias = -slopes_tensor.unsqueeze(-1).unsqueeze(-1) * distances.unsqueeze(0)

        return bias  # [num_heads, max_seq_len, max_seq_len]

    def _get_alibi_bias_for_length(self, seq_len: int) -> torch.Tensor:
        """Extract ALiBi bias for the given sequence length.

        Args:
            seq_len: Current sequence length

        Returns:
            Bias tensor [num_heads, seq_len, seq_len]
        """
        bias_tensor = cast(torch.Tensor, self.alibi_bias)
        if seq_len <= self.max_seq_len:
            return bias_tensor[:, :seq_len, :seq_len]

        slopes_tensor = cast(torch.Tensor, self.slopes)
        device = bias_tensor.device
        dtype = bias_tensor.dtype

        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        distances = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1)).to(
            dtype=dtype
        )

        bias = -slopes_tensor.to(device=device, dtype=dtype).unsqueeze(-1).unsqueeze(
            -1
        ) * distances.unsqueeze(0)
        return bias

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of ALiBi multi-head self-attention.

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

        # Apply linear projections
        q = self.w_q(x)  # [batch_size, seq_len, d_model]
        k = self.w_k(x)  # [batch_size, seq_len, d_model]
        v = self.w_v(x)  # [batch_size, seq_len, d_model]

        # Reshape for multi-head attention
        q = reshape_for_attention(q, self.num_heads, self.d_head)
        k = reshape_for_attention(k, self.num_heads, self.d_head)
        v = reshape_for_attention(v, self.num_heads, self.d_head)

        # Get ALiBi bias for current sequence length
        alibi_bias = self._get_alibi_bias_for_length(
            seq_len
        )  # [num_heads, seq_len, seq_len]

        # Ensure alibi_bias is on the same device as input
        alibi_bias = alibi_bias.to(device=x.device, dtype=x.dtype)

        # Expand ALiBi bias for batch dimension
        alibi_bias = alibi_bias.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Combine ALiBi bias with user-provided mask if given
        if mask is not None:
            mask_rank = mask.dim()
            if mask_rank == MASK_RANK_SQUARE:
                mask = (
                    mask.unsqueeze(0)
                    .unsqueeze(0)
                    .expand(batch_size, self.num_heads, -1, -1)
                )
            elif mask_rank == MASK_RANK_BATCH:
                mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

            # Convert boolean mask to additive mask if needed
            if mask.dtype == torch.bool:
                additive_mask = torch.where(mask, 0.0, float("-inf"))
            else:
                additive_mask = mask

            # Combine ALiBi bias with user mask
            combined_bias = alibi_bias + additive_mask
        else:
            combined_bias = alibi_bias

        # Compute scaled dot-product attention with ALiBi bias
        attention_output, attention_weights = scaled_dot_product_attention(
            q,
            k,
            v,
            mask=combined_bias,
            dropout=self.dropout,
            temperature=self.temperature,
        )

        # Reshape back to original format
        attention_output = reshape_from_attention(attention_output, self.d_model)

        # Apply output projection
        output = self.w_o(attention_output)

        # Store attention weights for visualization (average across heads)
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
                "max_seq_len": self.max_seq_len,
            }
        )
        return config

    def extra_repr(self) -> str:
        """Extra representation for debugging."""
        base_repr = super().extra_repr()
        return (
            f"{base_repr}, num_heads={self.num_heads}, d_head={self.d_head}, "
            f"temperature={self.temperature}, max_seq_len={self.max_seq_len}"
        )
