from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional

from attzoo.base import BaseSelfAttention


MASK_RANK_VECTOR = 1
MASK_RANK_MATRIX = 2
MASK_RANK_BATCH_MATRIX = 3


class LinearSelfAttention(BaseSelfAttention):
    """Linear self-attention implementation with O(n) complexity.

    This class implements linear attention using feature maps to approximate
    the softmax operation, achieving linear complexity instead of quadratic.
    Uses ELU activation with added constant for feature mapping.

    Reference: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
    by Katharopoulos et al. (2020)

    Args:
        d_model: Model dimension (dimension for Q, K, V projections and output)
        input_dim: Input feature dimension (default: None, uses d_model)
        dropout: Dropout probability (default: 0.1)
        bias: Whether to use bias in linear layers (default: False)
        feature_dim: Dimension of feature mapping (default: None, uses d_model)
        eps: Small constant for numerical stability (default: 1e-6)
        rope: Whether to apply Rotary Position Embedding (default: False)
    """

    def __init__(
        self,
        d_model: int,
        input_dim: int | None = None,
        dropout: float = 0.1,
        *,
        bias: bool = False,
        feature_dim: int | None = None,
        eps: float = 1e-6,
        rope: bool = False,
    ):
        super().__init__(d_model, input_dim, dropout, bias=bias, rope=rope)

        self.feature_dim = feature_dim if feature_dim is not None else d_model
        self.eps = eps

        # Linear projections for query, key, value, and output
        self.w_q = nn.Linear(self.input_dim, self.feature_dim, bias=bias)
        self.w_k = nn.Linear(self.input_dim, self.feature_dim, bias=bias)
        self.w_v = nn.Linear(self.input_dim, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

        # Initialize weights
        self._init_weights()

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feature mapping function to approximate softmax.

        Uses ELU activation with added constant as in the original paper.

        Args:
            x: Input tensor

        Returns:
            Feature mapped tensor
        """
        return functional.elu(x) + 1.0

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of linear self-attention.

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask (currently not supported for linear attention)
            **kwargs: Additional arguments

        Returns:
            Tuple of (output, attention_weights) where output is [batch_size, seq_len, d_model]
            Note: attention_weights are computed for compatibility but are approximate
        """
        del kwargs
        batch_size, seq_len, _ = x.shape

        q, k, v = self._project_inputs(x)
        q, k = self._maybe_apply_rope(q, k)
        q_prime, k_prime = self._feature_map_pair(q, k)

        sequence_mask = self._prepare_mask(mask, batch_size, seq_len)
        if sequence_mask is not None:
            k_prime, v = self._apply_sequence_mask(k_prime, v, sequence_mask)

        attention_output = self._compute_linear_attention(q_prime, k_prime, v)
        attention_output = self.dropout(attention_output)
        output = self.w_o(attention_output)

        attention_weights = self._estimate_attention_weights(q_prime, k_prime)
        self.attention_weights = attention_weights.detach()

        return output, attention_weights

    def _project_inputs(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        return q, k, v

    def _maybe_apply_rope(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.rope:
            return q, k
        q_with_head = q.unsqueeze(1)
        k_with_head = k.unsqueeze(1)
        q_rot, k_rot = self.apply_rope(q_with_head, k_with_head)
        return q_rot.squeeze(1), k_rot.squeeze(1)

    def _feature_map_pair(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._feature_map(q), self._feature_map(k)

    def _prepare_mask(
        self,
        mask: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor | None:
        if mask is None:
            return None
        mask_float = self._ensure_float_mask(mask)
        return self._to_sequence_mask(mask_float, batch_size, seq_len)

    @staticmethod
    def _ensure_float_mask(mask: torch.Tensor) -> torch.Tensor:
        return mask.float() if mask.dtype == torch.bool else mask

    def _to_sequence_mask(
        self, mask: torch.Tensor, batch_size: int, seq_len: int
    ) -> torch.Tensor:
        if (
            mask.dim() == MASK_RANK_MATRIX
            and mask.size(0) == seq_len
            and mask.size(1) == seq_len
        ):
            diagonal = mask.diagonal().unsqueeze(0).expand(batch_size, -1)
            return diagonal
        if mask.dim() == MASK_RANK_BATCH_MATRIX and mask.size(1) == mask.size(2):
            return mask.diagonal(dim1=1, dim2=2)
        if mask.dim() == MASK_RANK_MATRIX and mask.size(0) == batch_size:
            return mask
        if mask.dim() == MASK_RANK_VECTOR:
            return mask.unsqueeze(0).expand(batch_size, -1)
        message = f"Unsupported mask shape {mask.shape} for linear attention"
        raise ValueError(message)

    @staticmethod
    def _apply_sequence_mask(
        k_prime: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask_expanded = mask.unsqueeze(-1)
        return k_prime * mask_expanded, v * mask_expanded

    def _compute_linear_attention(
        self,
        q_prime: torch.Tensor,
        k_prime: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        kv = torch.einsum("bsf,bsd->bfd", k_prime, v)
        k_sum = k_prime.sum(dim=1)
        numerator = torch.einsum("bsf,bfd->bsd", q_prime, kv)
        denominator = (
            torch.einsum("bsf,bf->bs", q_prime, k_sum).unsqueeze(-1) + self.eps
        )
        return numerator / denominator

    def _estimate_attention_weights(
        self, q_prime: torch.Tensor, k_prime: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            q_norm = q_prime / (q_prime.sum(dim=-1, keepdim=True) + self.eps)
            k_norm = k_prime / (k_prime.sum(dim=-1, keepdim=True) + self.eps)
            attention_weights = torch.bmm(q_norm, k_norm.transpose(-2, -1))
            attention_weights = attention_weights / (
                attention_weights.sum(dim=-1, keepdim=True) + self.eps
            )
            return attention_weights

    def get_config(self) -> dict[str, Any]:
        """Get configuration dictionary."""
        config = super().get_config()
        config.update(
            {
                "feature_dim": self.feature_dim,
                "eps": self.eps,
            }
        )
        return config

    def extra_repr(self) -> str:
        """Extra representation for debugging."""
        base_repr = super().extra_repr()
        return f"{base_repr}, feature_dim={self.feature_dim}, eps={self.eps}"
