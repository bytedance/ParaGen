# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from paragen.modules.layers.feed_forward import FFN


from .axial_attention import ColumnSelfAttention, RowSelfAttention


class AxialTransformerLayer(nn.Module):
    """Implements an Axial MSA Transformer block."""

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        attention_dropout: float = 0,
        activation="relu",
        max_tokens_per_msa: int = 2 ** 14,
        normalize_before=True
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.d_model = d_model
        self.dropout_prob = dropout
        self.normalize_before = normalize_before

        row_self_attention = RowSelfAttention(
            d_model,
            nhead,
            dropout=attention_dropout,
            max_tokens_per_msa=max_tokens_per_msa,
        )

        column_self_attention = ColumnSelfAttention(
            d_model,
            nhead,
            dropout=attention_dropout,
            max_tokens_per_msa=max_tokens_per_msa,
        )

        feed_forward_layer = FFN(
            d_model,
            dim_feedforward=dim_feedforward,
            activation=activation,
       )

        self.row_self_attention = self.build_residual(row_self_attention)
        self.column_self_attention = self.build_residual(column_self_attention)
        self.feed_forward_layer = self.build_residual(feed_forward_layer)

    def build_residual(self, layer: nn.Module):
        return NormalizedResidualBlock(
            layer,
            self.d_model,
            self.dropout_prob,
            self.normalize_before
        )

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        need_head_weights: bool = False,
    ):
        r"""
        Args:
            src: tokens in src side.
              :math:`(G, S, B, E)` where G is the sequence number, S is the source sequence length, B is the batch size,
              E is the embedding size.
            src_key_padding_mask: the mask for the src keys per batch (optional).
                :math: `(B, G, S)`, where B is batch size, G is the sequence number and S is sequence length

        """
        x, row_attn = self.row_self_attention(
            x,
            self_attn_mask=self_attn_mask,
            self_attn_padding_mask=src_key_padding_mask,
        )
        x, column_attn = self.column_self_attention(
            x,
            self_attn_mask=self_attn_mask,
            self_attn_padding_mask=src_key_padding_mask,
        )
        x = self.feed_forward_layer(x)
        if need_head_weights:
            return x, column_attn, row_attn
        else:
            return x


class NormalizedResidualBlock(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        embedding_dim: int,
        dropout: float = 0.1,
        normalize_before = True
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.layer = layer
        self.dropout_module = nn.Dropout(
            dropout,
        )
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.normalize_before = normalize_before

    def forward(self, x, *args, **kwargs):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)
        outputs = self.layer(x, *args, **kwargs)
        if isinstance(outputs, tuple):
            x, *out = outputs
        else:
            x = outputs
            out = None

        x = self.dropout_module(x)
        x = residual + x

        if not self.normalize_before:
            x = self.layer_norm(x)
    
        if out is not None:
            return (x,) + tuple(out)
        else:
            return x