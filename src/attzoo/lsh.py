import math
from typing import Any

import torch
import torch.nn as nn

from attzoo.base import BaseSelfAttention, scaled_dot_product_attention
from attzoo.masks import expand_mask_for_heads
from attzoo.utils import reshape_for_attention, reshape_from_attention


class LSHSelfAttention(BaseSelfAttention):
    """LSH (Reformer-style) self-attention.

    This module approximates full attention by hashing tokens into buckets using
    random projections, then computing attention only within each bucket. Multiple
    independent hash rounds can be used and averaged to improve recall.

    Args:
        d_model: Model dimension (must be divisible by num_heads)
        num_heads: Number of attention heads
        input_dim: Input feature dimension (default: None, uses d_model)
        bucket_size: Target bucket size; number of buckets ~= ceil(seq_len / bucket_size)
        num_hashes: Number of independent hashing rounds to average (default: 1)
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
        bucket_size: int = 64,
        num_hashes: int = 1,
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
        if bucket_size <= 0:
            message = f"bucket_size must be positive, got {bucket_size}"
            raise ValueError(message)
        if num_hashes <= 0:
            message = f"num_hashes must be positive, got {num_hashes}"
            raise ValueError(message)

        super().__init__(d_model, input_dim, dropout, bias=bias, rope=rope)

        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.bucket_size = bucket_size
        self.num_hashes = num_hashes
        self.temperature = temperature

        # Projections
        self.w_q = nn.Linear(self.input_dim, d_model, bias=bias)
        self.w_k = nn.Linear(self.input_dim, d_model, bias=bias)
        self.w_v = nn.Linear(self.input_dim, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

        self._init_weights()

    def _compute_num_buckets(self, seq_len: int) -> int:
        # At least 1 bucket; prefer smaller number of buckets when seq is short
        return max(1, math.ceil(seq_len / self.bucket_size))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for LSH self-attention.

        Args:
            x: Input tensor [batch, seq, input_dim]
            mask: Optional attention mask. Supports same formats as MHSA:
                  [seq, seq], [batch, seq, seq], or [batch, heads, seq, seq],
                  with either boolean (True=attend) or additive (-inf for masked).

        Returns:
            Tuple of (output, attention_weights)
            - output: [batch, seq, d_model]
            - attention_weights: [batch, num_heads, seq, seq] (sparse; zeros outside buckets)
        """
        batch_size, seq_len, _ = x.shape
        q, k, v = self._project_inputs(x)
        q, k, v = self._reshape_to_heads(q, k, v)
        if self.rope:
            q, k = self.apply_rope(q, k)

        expanded_mask = expand_mask_for_heads(mask, batch_size, self.num_heads, seq_len)
        n_buckets = self._compute_num_buckets(seq_len)

        out_accum, attn_weights_accum = self._run_hash_rounds(
            q,
            k,
            v,
            expanded_mask,
            batch_size,
            seq_len,
            n_buckets,
            x.device,
            x.dtype,
        )

        out_accum = out_accum / float(self.num_hashes)
        attn_weights_accum = attn_weights_accum / float(self.num_hashes)

        out_cat = reshape_from_attention(out_accum, self.d_model)
        output = self.w_o(out_cat)
        self.attention_weights = attn_weights_accum.mean(dim=1).detach()

        return output, attn_weights_accum

    def _project_inputs(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.w_q(x), self.w_k(x), self.w_v(x)

    def _reshape_to_heads(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            reshape_for_attention(q, self.num_heads, self.d_head),
            reshape_for_attention(k, self.num_heads, self.d_head),
            reshape_for_attention(v, self.num_heads, self.d_head),
        )

    def _run_hash_rounds(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        expanded_mask: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
        n_buckets: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_accum = torch.zeros_like(q)
        weights_accum = q.new_zeros((batch_size, self.num_heads, seq_len, seq_len))

        for _ in range(self.num_hashes):
            projection = self._sample_projection(device, dtype, n_buckets)
            round_output, round_weights = self._process_hash_round(
                q,
                k,
                v,
                expanded_mask,
                projection,
                batch_size,
                seq_len,
                n_buckets,
            )
            output_accum += round_output
            weights_accum += round_weights
        return output_accum, weights_accum

    def _sample_projection(
        self, device: torch.device, dtype: torch.dtype, n_buckets: int
    ) -> torch.Tensor:
        return torch.randn(
            self.num_heads,
            self.d_head,
            n_buckets,
            device=device,
            dtype=dtype,
        )

    def _process_hash_round(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        expanded_mask: torch.Tensor | None,
        projection: torch.Tensor,
        batch_size: int,
        seq_len: int,
        n_buckets: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bucket_ids = self._assign_buckets(q, projection)
        round_output = torch.zeros_like(q)
        round_weights = q.new_zeros((batch_size, self.num_heads, seq_len, seq_len))

        for batch_index in range(batch_size):
            for head_index in range(self.num_heads):
                base_mask = self._select_head_mask(
                    expanded_mask, batch_index, head_index
                )
                self._process_single_head(
                    q,
                    k,
                    v,
                    bucket_ids,
                    round_output,
                    round_weights,
                    batch_index,
                    head_index,
                    base_mask,
                    n_buckets,
                )
        return round_output, round_weights

    @staticmethod
    def _assign_buckets(q: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
        scores = torch.einsum("bhld,hdm->bhlm", q, projection)
        return scores.argmax(dim=-1)

    @staticmethod
    def _select_head_mask(
        expanded_mask: torch.Tensor | None, batch_index: int, head_index: int
    ) -> torch.Tensor | None:
        if expanded_mask is None:
            return None
        return expanded_mask[batch_index, head_index]

    def _process_single_head(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bucket_ids: torch.Tensor,
        round_output: torch.Tensor,
        round_weights: torch.Tensor,
        batch_index: int,
        head_index: int,
        base_mask: torch.Tensor | None,
        n_buckets: int,
    ) -> None:
        head_ids = bucket_ids[batch_index, head_index]
        for bucket in range(n_buckets):
            indices = (head_ids == bucket).nonzero(as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            group_output, group_weights = self._compute_group_attention(
                q[batch_index, head_index],
                k[batch_index, head_index],
                v[batch_index, head_index],
                indices,
                base_mask,
            )
            round_output[batch_index, head_index, indices] += group_output
            self._scatter_attention_weights(
                round_weights[batch_index, head_index],
                indices,
                group_weights,
            )

    def _compute_group_attention(
        self,
        q_head: torch.Tensor,
        k_head: torch.Tensor,
        v_head: torch.Tensor,
        indices: torch.Tensor,
        base_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_group = q_head[indices]
        k_group = k_head[indices]
        v_group = v_head[indices]

        group_mask = None
        if base_mask is not None:
            group_mask = base_mask.index_select(0, indices).index_select(1, indices)

        out_g, attn_g = scaled_dot_product_attention(
            q_group.unsqueeze(0),
            k_group.unsqueeze(0),
            v_group.unsqueeze(0),
            mask=group_mask.unsqueeze(0) if group_mask is not None else None,
            dropout=self.dropout,
            temperature=self.temperature,
        )
        return out_g.squeeze(0), attn_g.squeeze(0)

    @staticmethod
    def _scatter_attention_weights(
        target: torch.Tensor, indices: torch.Tensor, group_weights: torch.Tensor
    ) -> None:
        ii = indices.view(-1, 1).expand(group_weights.size(0), group_weights.size(1))
        jj = indices.view(1, -1).expand(group_weights.size(0), group_weights.size(1))
        target.index_put_((ii, jj), group_weights, accumulate=True)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "d_head": self.d_head,
                "bucket_size": self.bucket_size,
                "num_hashes": self.num_hashes,
                "temperature": self.temperature,
            }
        )
        return config

    def extra_repr(self) -> str:
        base = super().extra_repr()
        return (
            f"{base}, num_heads={self.num_heads}, d_head={self.d_head}, "
            f"bucket_size={self.bucket_size}, num_hashes={self.num_hashes}, "
            f"temperature={self.temperature}"
        )
