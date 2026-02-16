"""Attention mechanisms for transformer models.

This module provides a comprehensive implementation of various attention mechanisms
including self-attention, multi-head attention, and cross-attention patterns.
All implementations follow the architectural patterns established for modularity
and extensibility.
"""

import math
from abc import ABC
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Dropout | None = None,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    d_k = query.size(-1)

    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / (math.sqrt(d_k) * temperature)

    # Apply mask if provided
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        else:
            scores = scores + mask

    # Apply softmax to get attention weights
    attention_weights = functional.softmax(scores, dim=-1)

    # Apply dropout if provided
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # Apply attention to values
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


class BaseSelfAttention(nn.Module, ABC):
    """Base class for self-attention mechanisms.

    Self-attention computes relationships within a single input sequence by
    using the same tensor for queries, keys, and values. This allows each
    position to attend to all positions in the same sequence.

    Args:
        d_model: Model dimension for attention computation
        input_dim: Input feature dimension (default: None, uses d_model)
        dropout: Dropout probability (default: 0.1)
        bias: Whether to use bias in linear layers (default: False)
        rope: Whether to apply Rotary Position Embedding (default: False)
    """

    def __init__(
        self,
        d_model: int,
        input_dim: int | None = None,
        dropout: float = 0.1,
        *,
        bias: bool = False,
        rope: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim if input_dim is not None else d_model
        self.dropout_prob = dropout
        self.bias = bias
        self.rope = rope
        self.dropout = nn.Dropout(dropout)

        # Initialize attention weights storage
        self.attention_weights: torch.Tensor | None = None

        # Initialize RoPE cache if enabled
        if self.rope:
            self._rope_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def _init_weights(self) -> None:
        """Initialize linear layer weights using Xavier uniform initialization.

        This method initializes the standard attention projection layers (w_q, w_k, w_v, w_o)
        using Xavier uniform initialization for weights and zeros for biases.
        Subclasses with these standard layers can call this method after creating them.
        """
        layer_names = ("w_q", "w_k", "w_v", "w_o")
        linear_layers = []
        for layer_name in layer_names:
            candidate = getattr(self, layer_name, None)
            if isinstance(candidate, nn.Linear):
                linear_layers.append(candidate)

        for module in linear_layers:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _create_projection_layers(
        self,
        q_dim: int | None = None,
        k_dim: int | None = None,
        v_dim: int | None = None,
        o_dim: int | None = None,
        *,
        bias: bool | None = None,
    ) -> tuple[nn.Linear, nn.Linear, nn.Linear, nn.Linear]:
        """Create standard Q, K, V, and output projection layers.

        This helper method creates the standard linear projection layers used by most
        attention mechanisms, reducing code duplication.

        Args:
            q_dim: Query projection output dimension (default: d_model)
            k_dim: Key projection output dimension (default: d_model)
            v_dim: Value projection output dimension (default: d_model)
            o_dim: Output projection output dimension (default: d_model)
            bias: Whether to use bias in layers (default: self.bias)

        Returns:
            Tuple of (w_q, w_k, w_v, w_o) linear layers
        """
        bias = self.bias if bias is None else bias
        default_dim = self.d_model
        q_dim = q_dim if q_dim is not None else default_dim
        k_dim = k_dim if k_dim is not None else default_dim
        v_dim = v_dim if v_dim is not None else default_dim
        o_dim = o_dim if o_dim is not None else default_dim

        w_q = nn.Linear(self.input_dim, q_dim, bias=bias)
        w_k = nn.Linear(self.input_dim, k_dim, bias=bias)
        w_v = nn.Linear(self.input_dim, v_dim, bias=bias)
        w_o = nn.Linear(self.d_model, o_dim, bias=bias)

        return w_q, w_k, w_v, w_o

    def apply_rope(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply Rotary Position Embedding (RoPE) to query and key tensors.

        Args:
            query: Query tensor [..., seq_len, d_head]
            key: Key tensor [..., seq_len, d_head]

        Returns:
            Tuple of (rotated_query, rotated_key)
        """
        if not self.rope:
            return query, key

        seq_len = query.size(-2)
        d_head = query.size(-1)

        # Get or compute RoPE frequencies
        cos, sin = self._get_rope_freqs(seq_len, d_head, query.device, query.dtype)

        # Apply RoPE to query and key
        query_rot = self._apply_rope_rotation(query, cos, sin)
        key_rot = self._apply_rope_rotation(key, cos, sin)

        return query_rot, key_rot

    def _get_rope_freqs(
        self, seq_len: int, d_head: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get or compute RoPE frequency tensors.

        Args:
            seq_len: Sequence length
            d_head: Head dimension
            device: Target device
            dtype: Target dtype

        Returns:
            Tuple of (cos, sin) frequency tensors
        """
        # Check cache
        cache_key = seq_len
        if cache_key in self._rope_cache:
            cos_cached, sin_cached = self._rope_cache[cache_key]
            if (
                cos_cached.device == device
                and cos_cached.dtype == dtype
                and cos_cached.size(-1) == d_head
            ):
                return cos_cached, sin_cached

        # Compute RoPE frequencies
        # theta_i = 10000^(-2i/d) for i in [0, 1, ..., d/2-1]
        dim_pairs = d_head // 2
        freqs = 1.0 / (
            10000.0
            ** (torch.arange(0, dim_pairs, dtype=dtype, device=device) * 2.0 / d_head)
        )

        # Create position indices
        positions = torch.arange(seq_len, dtype=dtype, device=device)

        # Outer product to get all position-frequency combinations
        angles = torch.outer(positions, freqs)  # [seq_len, d_head//2]

        # Compute cos and sin
        cos = torch.cos(angles)  # [seq_len, d_head//2]
        sin = torch.sin(angles)  # [seq_len, d_head//2]

        # Expand to match head dimension by repeating each frequency
        cos = torch.repeat_interleave(cos, 2, dim=-1)  # [seq_len, d_head]
        sin = torch.repeat_interleave(sin, 2, dim=-1)  # [seq_len, d_head]

        # Add dimensions for broadcasting: [1, seq_len, d_head]
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

        # Cache the result
        self._rope_cache[cache_key] = (cos, sin)

        return cos, sin

    def _apply_rope_rotation(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Apply RoPE rotation to input tensor.

        Args:
            x: Input tensor [..., seq_len, d_head]
            cos: Cosine frequencies [1, seq_len, d_head]
            sin: Sine frequencies [1, seq_len, d_head]

        Returns:
            Rotated tensor with same shape as input
        """
        # Split into even and odd dimensions
        x1 = x[..., ::2]  # Even dimensions
        x2 = x[..., 1::2]  # Odd dimensions

        # Apply rotation: [cos, -sin; sin, cos] @ [x1; x2]
        cos_part = cos[..., ::2]  # Even positions in cos
        sin_part = sin[
            ..., ::2
        ]  # Even positions in sin (same as cos for our implementation)

        # Rotary transformation
        rotated_x1 = x1 * cos_part - x2 * sin_part
        rotated_x2 = x1 * sin_part + x2 * cos_part

        # Interleave back to original format
        rotated = torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)

        return rotated

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for self-attention.

        Args:
            x: Value tensor (should be same as query for self-attention)
            mask: Optional attention mask
            **kwargs: Additional arguments

        Returns:
            Tuple of (output, attention_weights)
        """
        # This is an abstract implementation that should be overridden
        raise NotImplementedError("Subclasses must implement the forward method")

    def get_attention_weights(self) -> torch.Tensor:
        """Get attention weights from the most recent forward pass.

        Returns:
            Attention weights tensor

        Raises:
            RuntimeError: If no forward pass has been performed
        """
        if self.attention_weights is None:
            message = "No attention weights available. Perform a forward pass first."
            raise RuntimeError(message)
        return self.attention_weights

    def get_config(self) -> dict[str, Any]:
        """Get configuration dictionary for serialization.
        Returns:
            Configuration dictionary
        """
        return {
            "d_model": self.d_model,
            "input_dim": self.input_dim,
            "dropout": self.dropout_prob,
            "bias": self.bias,
            "rope": self.rope,
        }

    def extra_repr(self) -> str:
        """Extra representation for debugging."""
        return f"d_model={self.d_model}, input_dim={self.input_dim}, dropout={self.dropout_prob}, bias={self.bias}, rope={self.rope}"
