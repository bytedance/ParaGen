from typing import Tuple

from torch import Tensor
import torch
import torch.nn as nn

from paragen.modules.encoders import AbstractEncoder, register_encoder, create_encoder
from paragen.modules.encoders.key_value_transformer_encoder import KeyValueTransformerEncoder
from paragen.modules.encoders.layers.transformer_encoder_layer import TransformerEncoderLayer
from paragen.utils.ops import get_ordered_values_from_table_by_key
from paragen.utils.io import UniIO


@register_encoder
class MultiEncoderWrapper(AbstractEncoder):
    """
    MultiEncoderWrapper consists of multiple encoders and an upside transformer encoder, and compute representation
    from various sources.

    Args:
        encoders: encoder configurations
        num_layers: number of upside encoder layers
        d_model: feature dimension
        n_head: head numbers of multihead attention
        dim_feedforward: dimensionality of inner vector space
        dropout: dropout rate
        activation: activation function used in feed-forward network
        normalize_before: use pre-norm fashion, default as post-norm.
            Pre-norm suit deep nets while post-norm achieve better results when nets are shallow.
        name: module name
    """

    def __init__(self,
                 encoders,
                 num_layers,
                 d_model,
                 n_head,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 normalize_before=False,
                 name=None):
        super().__init__()
        self._encoders_configs = encoders
        self._num_layers = num_layers
        self._d_model = d_model
        self._n_head = n_head
        self._dim_feedforward = dim_feedforward
        self._dropout = dropout
        self._activation = activation
        self._normalize_before = normalize_before
        self._name = name
        self._embed_scale = d_model ** .5

        self._padding_idx = None
        self._encoders = None
        self._layers, self._norm = None, None

    def build(self, embed, special_tokens):
        """
        Build computational modules.

        Args:
            embed: token embedding
            special_tokens: special tokens defined in vocabulary
        """
        self._build_encoders(embed, special_tokens)
        self._layers = nn.ModuleList([TransformerEncoderLayer(d_model=self._d_model,
                                                              nhead=self._n_head,
                                                              dim_feedforward=self._dim_feedforward,
                                                              dropout=self._dropout,
                                                              activation=self._activation,
                                                              normalize_before=self._normalize_before)
                                      for _ in range(self._num_layers)])
        self._norm = nn.LayerNorm(self._d_model)

    def _build_encoders(self, embed, special_tokens):
        """
        Build multiple encoders for each source of input

        Args:
            embed: token embedding
            special_tokens: special tokens defined in vocabulary
        """
        self._encoders = nn.ModuleList()
        encoders, kv_encoder = {}, None
        for name, configs in self._encoders_configs.items():
            encoder = create_encoder(configs)
            encoder.build(embed, special_tokens)
            if isinstance(encoder, KeyValueTransformerEncoder):
                kv_encoder = encoder
            else:
                encoders[name] = encoder
        encoders = get_ordered_values_from_table_by_key(encoders)
        for encoder in encoders:
            self._encoders.append(encoder)
        if kv_encoder is not None:
            self._encoders.append(kv_encoder)

    def _forward(self, *args) -> Tuple[Tensor, Tensor]:
        r"""
        Args:
            *args: tokens in src side. By default, key-value pairs are placed at last position

        Returns:
            - source token hidden representation.
              :math:`(S, N, E)` where S is the source sequence length, N is the batch size,
              E is the embedding size.
        """
        memory, memory_padding_mask = [], []
        for encoder in self._encoders:
            if isinstance(encoder, KeyValueTransformerEncoder):
                x, x_padding_mask = encoder(args[0], args[1])
                args = args[2:]
            else:
                x, x_padding_mask = encoder(args[0])
                args = args[1:]
            memory.append(x)
            memory_padding_mask.append(x_padding_mask)
        x = torch.cat(memory, dim=0)
        x_padding_mask = torch.cat(memory_padding_mask, dim=1)

        for layer in self._layers:
            x = layer(x, src_key_padding_mask=x_padding_mask)

        x = self._norm(x)

        return x, x_padding_mask

    def save(self, path):
        """
        save module to path

        Args:
            path: saving path
        """
        for name, encoder in self._encoders.items():
            with UniIO('{}/{}.pt'.format(path, name), 'wb') as fout:
                torch.save(encoder, fout)
        with UniIO('{}/wrapper.pt'.format(path), 'wb') as fout:
            torch.save({'layers': self._layers, 'norm': self._norm}, fout)

    @property
    def d_model(self):
        return self._d_model

    @property
    def out_dim(self):
        return self._d_model

    def reset(self, mode):
        """
        Reset encoder and switch running mode

        Args:
            mode: running mode in [train, valid, infer]
        """
        self._cache.clear()
        for encoder in self._encoders:
            encoder.reset(mode)
