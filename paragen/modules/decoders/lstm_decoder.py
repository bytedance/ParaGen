import torch.nn as nn
import torch.nn.functional as F

from paragen.modules.decoders import AbstractDecoder, register_decoder
from paragen.modules.decoders.layers.lstm_decoder_layer import LSTMDecoderLayer


@register_decoder
class LSTMDecoder(AbstractDecoder):
    """
    LSTMEncoder is a LSTM encoder.

    Args:
        num_layers: number of encoder layers
        d_model: feature dimension
        n_head: head numbers of multihead attention
        hidden_size: hidden size within LSTM
        dropout: dropout rate
        activation: activation function used in feed-forward network
        normalize_before: use pre-norm fashion, default as post-norm.
            Pre-norm suit deep nets while post-norm achieve better results when nets are shallow.
        name: module name
    """

    def __init__(self,
                 num_layers,
                 d_model,
                 n_head,
                 hidden_size=2048,
                 dropout=0.1,
                 activation='relu',
                 normalize_before=False,
                 name=None):
        super().__init__()
        self._num_layers = num_layers
        self._d_model = d_model
        self._n_head = n_head
        self._hidden_size = hidden_size
        self._dropout = dropout
        self._normalize_before = normalize_before
        self._activation = activation
        self._name = name
        self._embed_scale = d_model ** .5

        self._special_tokens = None
        self._embed, self._pos_embed, self._embed_dropout = None, None, None
        self._layers = None
        self._norm = None
        self._out_proj = None

    def build(self,
              embed,
              special_tokens,
              out_proj=None):
        """
        Build computational modules.

        Args:
            embed: token embedding
            special_tokens: special tokens defined in vocabulary
            out_proj: output projection. It is allowed to be initialized with embedding weight in model buildup.
        """
        self._embed = embed
        self._special_tokens = special_tokens
        self._embed_dropout = nn.Dropout(self._dropout)

        self._layers = nn.ModuleList([
            LSTMDecoderLayer(self._d_model,
                             self._n_head,
                             dim_feedforward=self._hidden_size,
                             dropout=self._dropout,
                             normalize_before=self._normalize_before,
                             activation=self._activation)
        ])

        self._norm = nn.LayerNorm(self._d_model) if self._normalize_before else None
        self._out_proj = out_proj

    def forward(self, tgt, memory, memory_padding_mask):
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
        x = self._embed(tgt) * self._embed_scale
        x = self._embed_dropout(x)

        x = x.transpose(0, 1)
        for i, layer in enumerate(self._layers):
            x = layer(x, memory, memory_padding_mask)
        x = x.transpose(0, 1)

        if self._norm is not None:
            x = self._norm(x)

        logits = self._out_proj(x)
        return logits

    def reset(self, mode):
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
        return {layer.get_cache() for i, layer in enumerate(self._layers)}

    def set_cache(self, cache):
        """
        Set cache from outside

        Args:
            cache: cache dict from outside
        """
        for i, layer in enumerate(self._layers):
            layer.set(cache[i])
