from typing import Optional

import torch
import torch.nn as nn

from paragen.modules.decoders import AbstractDecoder, register_decoder
from paragen.modules.layers.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from paragen.modules.layers.learned_positional_embedding import LearnedPositionalEmbedding
from paragen.modules.decoders.layers.transformer_decoder_layer import TransformerDecoderLayer
from paragen.modules.utils import create_time_mask


@register_decoder
class TransformerDecoder(AbstractDecoder):
    """
    TransformerEncoder is a transformer encoder.

    Args:
        num_layers: number of encoder layers
        d_model: feature dimension
        n_head: head numbers of multihead attention
        dim_feedforward: dimensionality of inner vector space
        dropout: dropout rate
        activation: activation function used in feed-forward network
        learn_pos: learning postional embedding instead of sinusoidal one
        normalize_before: use pre-norm fashion, default as post-norm.
            Pre-norm suit deep nets while post-norm achieve better results when nets are shallow.
        name: module name
    """

    def __init__(self,
                 num_layers,
                 d_model,
                 n_head,
                 dim_feedforward=2048,
                 dropout=0.1,
                 attention_dropout=0.,
                 activation='relu',
                 learn_pos=False,
                 normalize_before=False,
                 embed_scale=True,
                 embed_layer_norm=False,
                 output_bias=False,
                 max_pos=1024,
                 share_layers=False,
                 position_emb_post_mask=False,
                 name=None):
        super().__init__()
        self._num_layers = num_layers
        self._d_model = d_model
        self._n_head = n_head
        self._dim_feedforward = dim_feedforward
        self._dropout = dropout
        self._attention_dropout = attention_dropout
        self._activation = activation
        self._learn_pos = learn_pos
        self._normalize_before = normalize_before
        self._output_bias = output_bias
        self._name = name
        self._embed_scale = d_model ** .5 if embed_scale else None
        self._embed_layer_norm = embed_layer_norm
        self._max_pos = max_pos
        self._share_layers = share_layers

        self._special_tokens = None
        self._embed, self._embed_norm, self._pos_embed, self._embed_dropout = None, None, None, None
        self._layer, self._layers = None, None
        self._norm = None
        self._out_proj = None
        self._out_proj_bias = None
        self._position_emb_post_mask = position_emb_post_mask

    def build(self,
              embed,
              special_tokens,
              out_proj):
        """
        Build computational modules.

        Args:
            embed: token embedding
            special_tokens: special tokens defined in vocabulary
            out_proj: output projection. It is allowed to be initialized with embedding weight in model buildup.
        """
        self._embed = embed
        self._special_tokens = special_tokens
        if self._learn_pos:
            self._pos_embed = LearnedPositionalEmbedding(num_embeddings=self._max_pos,
                                                         embedding_dim=self._d_model,
                                                         padding_idx=special_tokens['pad'],
                                                         post_mask=self._position_emb_post_mask)
        else:
            self._pos_embed = SinusoidalPositionalEmbedding(self._d_model)
        self._embed_norm = nn.LayerNorm(self._d_model) if self._embed_layer_norm else None
        self._embed_dropout = nn.Dropout(self._dropout)
        if self._share_layers:
            self._layer = TransformerDecoderLayer(d_model=self._d_model,
                                                  nhead=self._n_head,
                                                  dim_feedforward=self._dim_feedforward,
                                                  dropout=self._dropout,
                                                  attention_dropout=self._attention_dropout,
                                                  activation=self._activation,
                                                  normalize_before=self._normalize_before)
            self._layers = [self._layer for _ in range(self._num_layers)]
        else:
            self._layers = nn.ModuleList([TransformerDecoderLayer(d_model=self._d_model,
                                                                  nhead=self._n_head,
                                                                  dim_feedforward=self._dim_feedforward,
                                                                  dropout=self._dropout,
                                                                  attention_dropout=self._attention_dropout,
                                                                  activation=self._activation,
                                                                  normalize_before=self._normalize_before)
                                          for _ in range(self._num_layers)])
        self._norm = nn.LayerNorm(self._d_model) if self._normalize_before else None
        self._out_proj = out_proj
        if self._output_bias:
            self._out_proj_bias = nn.Parameter(torch.zeros(out_proj.weight.size(0)))

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                memory_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""
        Args:
            tgt: previous tokens in tgt side.
              :math:`(N, L)` where N is the batch size, L is the target sequence length.
              E is the embedding dimension.
            memory: memory for attention.
              :math:`(M, N, E)`, where M is the memory sequence length, N is the batch size,
            memory_padding_mask: memory sequence padding mask.
              :math:`(N, M)` where M is the memory sequence length, N is the batch size.


        Returns:
            - estimated logits.
              :math:`(N, L, V)` where N is the batch size, L is the target sequence length,
              V is the vocabulary size.
        """
        x = self._embed(tgt)
        if self._embed_scale is not None:
            x = x * self._embed_scale
        if self._pos_embed is not None:
            x = x + self._pos_embed(tgt)
        if self._embed_norm is not None:
            x = self._embed_norm(x)
        x = self._embed_dropout(x)

        x = x.transpose(0, 1)

        tgt_mask = create_time_mask(tgt)
        tgt_padding_mask = tgt.eq(self._special_tokens['pad'])
        for layer in self._layers:
            x = layer(tgt=x,
                      memory=memory,
                      tgt_mask=tgt_mask,
                      tgt_key_padding_mask=tgt_padding_mask,
                      memory_key_padding_mask=memory_padding_mask,)
        if self._norm is not None:
            x = self._norm(x)
        x = x.transpose(0, 1)
        logits = self._out_proj(x)
        if self._out_proj_bias is not None:
            logits = logits + self._out_proj_bias
        return logits

    def reset(self, mode='train'):
        """
        Reset encoder and switch running mode

        Args:
            mode: running mode in [train, valid, infer]
        """
        self._mode = mode
        for layer in self._layers:
            layer.reset(mode)

    def get_cache(self):
        """
        Retrieve inner cache

        Returns:
            - cached states as a Dict
        """
        return {i: layer.get_cache() for i, layer in enumerate(self._layers)}

    def set_cache(self, cache):
        """
        Set cache from outside

        Args:
            cache: cache dict from outside
        """
        for i, layer in enumerate(self._layers):
            layer.set_cache(cache[i])
