import math

import torch
import torch.onnx.operators
from torch import nn


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    class __SinusoidalPositionalEmbedding(nn.Module):

        def __init__(self, embedding_dim, num_embeddings=1024):
            super().__init__()
            self._embedding_dim = embedding_dim
            self._num_embeddings = num_embeddings

            num_timescales = self._embedding_dim // 2
            log_timescale_increment = torch.FloatTensor([math.log(10000.) / (num_timescales - 1)])
            inv_timescales = nn.Parameter((torch.arange(num_timescales) * -log_timescale_increment).exp(), requires_grad=False)
            self.register_buffer('_inv_timescales', inv_timescales)

        def forward(
            self,
            input,
        ):
            """Input is expected to be of size [bsz x seqlen]."""
            mask = torch.ones_like(input).type_as(self._inv_timescales)
            positions = torch.cumsum(mask, dim=1) - 1

            scaled_time = positions[:, :, None] * self._inv_timescales[None, None, :]
            signal = torch.cat([scaled_time.sin(), scaled_time.cos()], dim=-1)
            return signal.detach()

    __embed__ = None

    def __init__(self, embedding_dim, num_embeddings=1024):
        super().__init__()
        if not SinusoidalPositionalEmbedding.__embed__:
            SinusoidalPositionalEmbedding.__embed__ = SinusoidalPositionalEmbedding.__SinusoidalPositionalEmbedding(
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings
            )
        self.embedding = SinusoidalPositionalEmbedding.__embed__

    def forward(self, input):
        return self.embedding(input)
