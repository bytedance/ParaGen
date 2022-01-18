from torch import Tensor
from torch.nn import LSTM
import torch.nn as nn

from paragen.modules.encoders import AbstractEncoder, register_encoder


@register_encoder
class LSTMEncoder(AbstractEncoder):
    """
    LSTMEncoder is a LSTM encoder.

    Args:
        num_layers: number of encoder layers
        d_model: feature dimension
        hidden_size: hidden size within LSTM
        dropout: dropout rate
        bidirectional: encode sequence with bidirectional LSTM
        return_seed: return with sequence representation
        name: module name
    """

    def __init__(self,
                 num_layers,
                 d_model=512,
                 hidden_size=1024,
                 dropout=0.1,
                 bidirectional=True,
                 return_seed=None,
                 name=None):
        super().__init__()
        self._num_layers = num_layers
        self._d_model = d_model
        self._hidden_size = hidden_size
        self._dropout = dropout
        self._bidirectional = bidirectional
        self._return_seed = return_seed
        self._name = name

        self._special_tokens = None
        self._embed, self._embed_dropout = None, None
        self._layer = None
        self._pool_seed = None

    def build(self, embed, special_tokens):
        """
        Build computational modules.

        Args:
            embed: token embedding
            special_tokens: special tokens defined in vocabulary
        """
        self._embed = embed
        self._special_tokens = special_tokens
        self._embed_dropout = nn.Dropout(self._dropout)
        self._layer = LSTM(input_size=self._d_model,
                           hidden_size=self._hidden_size,
                           num_layers=self._num_layers,
                           dropout=self._dropout,
                           bidirectional=self._bidirectional)

    def _forward(self,
                 src: Tensor):
        r"""
        Args:
            src: tokens in src side.
              :math:`(N, S)` where N is the batch size, S is the source sequence length.

        Returns:
            - source token hidden representation.
                  :math:`(S, N, E)` where S is the source sequence length, N is the batch size,
                  E is the embedding size.
        """
        x = self._embed(src)
        x = self._embed_dropout(x)

        src_padding_mask = src.eq(self._special_tokens['pad'])
        x = x.transpose(0, 1)
        x = self._layer(x)[0]

        if self._pool_seed:
            return x, src_padding_mask, x.mean(dim=0)
        else:
            return x, src_padding_mask

    @property
    def d_model(self):
        return self._d_model

    @property
    def out_dim(self):
        return self._hidden_size * 2 if self._bidirectional else self._hidden_size
