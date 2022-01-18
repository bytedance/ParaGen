from typing import Tuple

from torch import Tensor
import torch.nn as nn

from paragen.modules.encoders import AbstractEncoder, register_encoder
from paragen.modules.layers.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from paragen.modules.encoders.layers.transformer_encoder_layer import TransformerEncoderLayer
from paragen.modules.utils import fuse_key_value


@register_encoder
class KeyValueTransformerEncoder(AbstractEncoder):
    """
    KeyValueTransformerEncoder is an extension to TransformerEncoder to encode key-value or table data.

    Args:
        num_key_layers: number of encoder layers to encode keys
        num_value_layers: number of encoder layers to encode values
        d_model: feature dimension
        n_head: head numbers of multihead attention
        dim_feedforward: dimensionality of inner vector space
        dropout: dropout rate
        activation: activation function used in feed-forward network
        fusing_key_val: fusion methods of key and value
        normalize_before: use pre-norm fashion, default as post-norm.
            Pre-norm suit deep nets while post-norm achieve better results when nets are shallow.
        name: module name
    """

    def __init__(self,
                 num_key_layers,
                 num_value_layers,
                 d_model=512,
                 n_head=8,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 fusing_key_val='max-pool-value',
                 normalize_before=False,
                 name=None):
        super().__init__()
        self._num_key_layers = num_key_layers
        self._num_value_layers = num_value_layers
        self._d_model = d_model
        self._n_head = n_head
        self._dim_feedforward = dim_feedforward
        self._dropout = dropout
        self._fusing_key_val = fusing_key_val
        self._activation = activation
        self._normalize_before = normalize_before
        self._name = name
        self._embed_scale = d_model ** .5

        self._padding_idx = None
        self._embed, self._pos_embed, self._embed_dropout, self._norm = None, None, None, None
        self._key_layers, self._value_layers = None, None

    def build(self, embed, special_tokens):
        """
        Build computational modules.

        Args:
            embed: token embedding
            special_tokens: special tokens defined in vocabulary
        """
        self._embed = embed
        self._padding_idx = special_tokens['pad']
        self._pos_embed = SinusoidalPositionalEmbedding(self._d_model)
        self._embed_dropout = nn.Dropout(self._dropout)
        self._key_layers = nn.ModuleList([TransformerEncoderLayer(d_model=self._d_model,
                                                                  nhead=self._n_head,
                                                                  dim_feedforward=self._dim_feedforward,
                                                                  dropout=self._dropout,
                                                                  activation=self._activation,
                                                                  normalize_before=self._normalize_before)
                                          for _ in range(self._num_key_layers)])
        self._value_layers = nn.ModuleList([TransformerEncoderLayer(d_model=self._d_model,
                                                                    nhead=self._n_head,
                                                                    dim_feedforward=self._dim_feedforward,
                                                                    dropout=self._dropout,
                                                                    activation=self._activation,
                                                                    normalize_before=self._normalize_before)
                                            for _ in range(self._num_value_layers)])
        self._norm = nn.LayerNorm(self._d_model) if self._normalize_before else None

    def _forward(self,
                 key: Tensor,
                 value: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        Args:
            key: key tokens.
              :math:`(N, K, L)` where N is the batch size, K is the key number, L is the value length.
            value: value tokens.
              :math:`(N, K)` where N is the batch size, K is the key number.

        Returns:
            - source token hidden representation.
              :math:`(M, N, E)` where S is the memory size, N is the batch size,
              E is the embedding size.
        """
        batch_size, ksize, vsize = value.size()

        value = value.view(batch_size * ksize, vsize)
        x = self._embed(value) * self._embed_scale
        x = x + self._pos_embed(value)
        x = self._embed_dropout(x)

        value_padding_mask = value.eq(self._padding_idx)
        x = x.transpose(0, 1)
        for layer in self._value_layers:
            x = layer(x, src_key_padding_mask=value_padding_mask)
        x = x.view(vsize, batch_size, ksize, self._d_model).transpose(1, 2)

        y = self._embed(key) * self._embed_scale
        y = self._embed_dropout(y)

        y = y.transpose(0, 1)
        key_padding_mask = key.eq(self._padding_idx)

        x, memory_padding_mask = fuse_key_value(key=y,
                                                value=x,
                                                key_padding_mask=key_padding_mask,
                                                value_padding_mask=value_padding_mask,
                                                fusing=self._fusing_key_val)
        for layer in self._key_layers:
            x = layer(x, src_key_padding_mask=memory_padding_mask)

        if self._norm is not None:
            x = self._norm(x)

        return x, memory_padding_mask

    @property
    def d_model(self):
        return self._d_model

    @property
    def out_dim(self):
        return self._d_model
