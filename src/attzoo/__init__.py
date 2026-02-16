"""Attzoo: A PyTorch library for attention mechanisms.

This library provides clean, efficient implementations of various self-attention
mechanisms used in transformer models and deep learning architectures.
"""

from attzoo.alibi import AlibiSelfAttention
from attzoo.base import (
    BaseSelfAttention,
    scaled_dot_product_attention,
)
from attzoo.block import BlockSelfAttention
from attzoo.combined import CombinedAttention
from attzoo.dilation import DilatedSelfAttention
from attzoo.gated import GatedSelfAttention
from attzoo.group import GroupedSelfAttention
from attzoo.linear import LinearSelfAttention
from attzoo.local import LocalSelfAttention
from attzoo.longformer import LongformerSelfAttention
from attzoo.lsh import LSHSelfAttention
from attzoo.masks import (
    combine_masks,
    create_block_mask,
    create_causal_mask,
    create_dilated_mask,
    create_local_mask,
    create_padding_mask,
    expand_mask_for_heads,
)
from attzoo.mhsa import MultiHeadSelfAttention
from attzoo.vanilla import VanillaSelfAttention


__version__ = "0.1.0"

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
