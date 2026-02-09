"""Attentions: A PyTorch library for attention mechanisms.

This library provides clean, efficient implementations of various self-attention
mechanisms used in transformer models and deep learning architectures.
"""

from attentions.alibi import AlibiSelfAttention
from attentions.base import (
    BaseSelfAttention,
    scaled_dot_product_attention,
)
from attentions.block import BlockSelfAttention
from attentions.combined import CombinedAttention
from attentions.dilation import DilatedSelfAttention
from attentions.gated import GatedSelfAttention
from attentions.group import GroupedSelfAttention
from attentions.linear import LinearSelfAttention
from attentions.local import LocalSelfAttention
from attentions.longformer import LongformerSelfAttention
from attentions.lsh import LSHSelfAttention
from attentions.masks import (
    combine_masks,
    create_block_mask,
    create_causal_mask,
    create_dilated_mask,
    create_local_mask,
    create_padding_mask,
    expand_mask_for_heads,
)
from attentions.mhsa import MultiHeadSelfAttention
from attentions.vanilla import VanillaSelfAttention


__version__ = "0.1.01"

__all__ = (
    "AlibiSelfAttention",
    "BaseSelfAttention",
    "BlockSelfAttention",
    "CombinedAttention",
    "DilatedSelfAttention",
    "GatedSelfAttention",
    "GroupedSelfAttention",
    "LSHSelfAttention",
    "LinearSelfAttention",
    "LocalSelfAttention",
    "LongformerSelfAttention",
    "MultiHeadSelfAttention",
    "VanillaSelfAttention",
    "combine_masks",
    "create_block_mask",
    "create_causal_mask",
    "create_dilated_mask",
    "create_local_mask",
    "create_padding_mask",
    "expand_mask_for_heads",
    "scaled_dot_product_attention",
)
