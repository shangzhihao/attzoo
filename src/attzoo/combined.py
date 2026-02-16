from typing import Any

import torch
import torch.nn as nn

from attzoo.base import BaseSelfAttention


MULTI_HEAD_RANK = 4


def _make_att_weight(
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


class CombinedAttention(nn.Module):
    """Blend two self-attention modules with a learned gate.

    y = g ⊙ AttnA(x) + (1 - g) ⊙ AttnB(x), with g = sigmoid(G(x)).

    Stores head-averaged, gate-weighted attention weights for inspection via
    ``get_attention_weights()``. Returns raw weights from ``attn_a`` for API
    compatibility.
    """

    def __init__(
        self,
        attn_a: BaseSelfAttention,
        attn_b: BaseSelfAttention,
        *,
        input_dim: int | None = None,
        gate_hidden: int | None = None,
        gate_bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if attn_a.d_model != attn_b.d_model:
            message = (
                f"attn_a.d_model ({attn_a.d_model}) must equal "
                f"attn_b.d_model ({attn_b.d_model})"
            )
            raise ValueError(message)
        self.attn_a = attn_a
        self.attn_b = attn_b
        self.d_model = attn_a.d_model
        self.input_dim = input_dim if input_dim is not None else self.d_model
        self.gate = _make_att_weight(
            self.input_dim,
            gate_hidden=gate_hidden,
            gate_bias=gate_bias,
            dropout=dropout,
        )
        self.attention_weights: torch.Tensor | None = None

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out_a, w_a = self.attn_a(x, mask=mask, **kwargs)
        out_b, w_b = self.attn_b(x, mask=mask, **kwargs)

        g = self.gate(x)  # [B, S, 1]
        output = g * out_a + (1.0 - g) * out_b

        def _to_seq_weights(w: torch.Tensor) -> torch.Tensor:
            return w.mean(dim=1) if w.dim() == MULTI_HEAD_RANK else w

        w_a_seq = _to_seq_weights(w_a)
        w_b_seq = _to_seq_weights(w_b)
        blended_seq_weights = g * w_a_seq + (1.0 - g) * w_b_seq
        self.attention_weights = blended_seq_weights.detach()

        return output, w_a

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
