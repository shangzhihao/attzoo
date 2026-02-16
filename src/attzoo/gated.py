from typing import Any

import torch
import torch.nn as nn

from attzoo.base import BaseSelfAttention


MULTI_HEAD_RANK = 4


def _make_gate(
    input_dim: int,
    *,
    gate_hidden: int | None,
    gate_bias: bool,
    dropout: float,
) -> nn.Sequential:
    if gate_hidden is None:
        return nn.Sequential(
            nn.Linear(input_dim, 1, bias=gate_bias),
            nn.Sigmoid(),
        )
    return nn.Sequential(
        nn.Linear(input_dim, gate_hidden, bias=gate_bias),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(gate_hidden, 1, bias=gate_bias),
        nn.Sigmoid(),
    )


class GatedSelfAttention(nn.Module):
    """Residual-gated self-attention (highway-style gate).

    x̂ = g ⊙ Attn(x) + (1 - g) ⊙ Proj(x), with g = sigmoid(G(x)).

    If the input dimension differs from the attention's ``d_model``, an internal
    projection aligns ``x`` to ``d_model`` before mixing.
    """

    def __init__(
        self,
        attn: BaseSelfAttention,
        *,
        input_dim: int | None = None,
        gate_hidden: int | None = None,
        gate_bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = attn
        self.d_model = attn.d_model
        # Default to the attention's expected input dimension when available
        self.input_dim = (
            input_dim
            if input_dim is not None
            else getattr(attn, "input_dim", self.d_model)
        )

        # Projection for x to match d_model if needed
        self.x_proj: nn.Module
        if self.input_dim != self.d_model:
            self.x_proj = nn.Linear(self.input_dim, self.d_model, bias=True)
        else:
            self.x_proj = nn.Identity()

        # Per-token gate over the input features
        self.gate = _make_gate(
            self.input_dim,
            gate_hidden=gate_hidden,
            gate_bias=gate_bias,
            dropout=dropout,
        )

        self.attention_weights: torch.Tensor | None = None

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_w = self.attn(x, mask=mask, **kwargs)  # [B, S, d_model], weights
        gate = self.gate(x)  # [B, S, 1]
        x_res = self.x_proj(x)  # [B, S, d_model]

        y = gate * attn_out + (1.0 - gate) * x_res

        # Store convenient head-averaged weights
        if attn_w.dim() == MULTI_HEAD_RANK:
            self.attention_weights = attn_w.mean(dim=1).detach()
        else:
            self.attention_weights = attn_w.detach()

        return y, attn_w

    def get_attention_weights(self) -> torch.Tensor:
        if self.attention_weights is None:
            message = "No attention weights available. Perform a forward pass first."
            raise RuntimeError(message)
        return self.attention_weights

    def get_config(self) -> dict[str, Any]:
        return {
            "d_model": self.d_model,
            "input_dim": self.input_dim,
        }

    def extra_repr(self) -> str:  # pragma: no cover - representation helper
        return f"d_model={self.d_model}, input_dim={self.input_dim}"
