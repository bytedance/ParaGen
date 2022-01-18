from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.

    Args:
        num_embeddings: number of embeddings
        embedding_dim: embedding dimension
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = None, post_mask=False):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        nn.init.normal_(self.weight, mean=0, std=embedding_dim ** -0.5)
        # if post_mask = True, then padding_idx = id of pad token in token embedding, we first mark padding
        # tokens using padding_idx, then generate embedding matrix using positional embedding, finally set
        # marked positions with zero
        self._post_mask = post_mask

    def forward(
        self,
        input: Tensor,
        positions: Optional[Tensor] = None
    ):
        """
        Args:
             input: an input LongTensor
                :math:`(*, L)`, where L is sequence length
            positions: pre-defined positions
                :math:`(*, L)`, where L is sequence length

        Returns:
            - positional embedding indexed from input
                :math:`(*, L, D)`, where L is sequence length and D is dimensionality
        """
        if self._post_mask:
            mask = input.ne(self.padding_idx).long()
            if positions is None:
                positions = (torch.cumsum(mask, dim=1) - 1).long()
            emb = F.embedding(
                positions,
                self.weight,
                None,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )#[B,L,H]
            emb = emb * mask.unsqueeze(-1)
            return emb
        else:
            if positions is None:
                mask = torch.ones_like(input)
                positions = (torch.cumsum(mask, dim=1) - 1).long()
            return F.embedding(
                positions,
                self.weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
