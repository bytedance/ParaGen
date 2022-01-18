from typing import Optional

from torch import Tensor
import torch
import torch.nn as nn

from paragen.modules.decoders.layers import AbstractDecoderLayer
from paragen.modules.layers.feed_forward import FFN


class LSTMDecoderLayer(AbstractDecoderLayer):
    """
    LSTMDecoderLayer performs LSTM decoding on sequence,
    namely LSTM, decoder-encoder multihead-attention and feed-forward network.

    Args:
        d_model: feature dimension
        nhead: head numbers of multihead attention
        dim_feedforward: dimensionality of inner vector space
        dropout: dropout rate
        activation: activation function used in feed-forward network
        normalize_before: use pre-norm fashion, default as post-norm.
            Pre-norm suit deep nets while post-norm achieve better results when nets are shallow.
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False,):
        super(LSTMDecoderLayer, self).__init__()
        self._d_model = d_model

        self._normalize_before = normalize_before
        self.rnn = nn.LSTM(d_model, d_model, 1)
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ffn = FFN(d_model, dim_feedforward, activation=activation)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self,
                tgt: Tensor,
                memory: Tensor,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
                :math:`(T, B, D)`, where T is sequence length, B is batch size and D is feature dimension
            memory: the sequence from the last layer of the encoder (required).
                :math:`(M, B, D)`, where M is memory size, B is batch size and D is feature dimension
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
                :math: `(B, M)`, where B is batch size and M is memory size.
        """
        if self._mode == 'infer':
            tgt = tgt[-1:]
        residual = tgt
        if self._normalize_before:
            tgt = self.norm1(tgt)
        prev_states = getattr(self._cache, 'prev_states', self._create_initial_seed(tgt))
        tgt, states = self.rnn(tgt, prev_states)
        if self._mode == 'infer':
            self._update_cache(states)
        tgt = self.dropout1(tgt)
        tgt = tgt + residual
        if not self._normalize_before:
            tgt = self.norm1(tgt)

        residual = tgt
        if self._normalize_before:
            tgt = self.norm2(tgt)
        tgt = self.attention(tgt, memory, memory,
                             key_padding_mask=memory_key_padding_mask)[0]
        tgt = self.dropout2(tgt)
        tgt = tgt + residual
        if not self._normalize_before:
            tgt = self.norm2(tgt)

        residual = tgt
        if self._normalize_before:
            tgt = self.norm3(tgt)
        tgt = self.ffn(tgt)
        tgt = self.dropout3(tgt)
        tgt = residual + tgt
        if not self._normalize_before:
            tgt = self.norm3(tgt)
        return tgt

    def _update_cache(self, states):
        """
        Update cache with current states

        Args:
            states: current state
        """
        self._cache['prev_states'] = states

    def _create_initial_seed(self, tgt):
        """
        Create initial seed (zeros) for LSTM encoder

        Args:
            tgt: target sequence
                :math:`(T, B, *), where T is sequence length and B is batch size

        Returns:
            - LSTM initial cell states
                :math:`(B, *)`, where B is batch size
            - LSTM initial output states
                :math:`(B, *)`, where B is batch size
        """
        batch_size = tgt.size(1)
        return torch.zeros(1, batch_size, self._d_model), torch.zeros(1, batch_size, self._d_model)
