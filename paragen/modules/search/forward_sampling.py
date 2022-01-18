from typing import Optional

import torch
import torch.nn.functional as F
import torch.distributions as D

from paragen.modules.search import register_search
from paragen.modules.search.sequence_search import SequenceSearch
from paragen.modules.utils import create_init_scores


@register_search
class ForwardSampling(SequenceSearch):
    """
    ForwardSampling is a sampling method in sequnece generation

    Args:
        maxlen_coef (a, b): maxlen computation coefficient.
            The max length is computed as `(S * a + b)`, where S is source sequence length.
        topk: sample next token from top-k ones
    """

    def __init__(self, maxlen_coef=(1.2, 10), topk=0):
        super().__init__()

        self._maxlen_a, self._maxlen_b = maxlen_coef
        self._topk = topk

    def forward(self,
                tokens,
                memory,
                memory_padding_mask,
                target_mask: Optional[torch.Tensor] = None,
                prev_scores: Optional[torch.Tensor] = None):
        """
        Decoding full-step sequence via sampling

        Args:
            tokens: previous tokens or prefix of sequence
            memory: memory for attention.
              :math:`(M, N, E)`, where M is the memory sequence length, N is the batch size,
            memory_padding_mask: memory sequence padding mask.
              :math:`(N, M)` where M is the memory sequence length, N is the batch size.
            target_mask: target mask indicating blacklist tokens
              :math:`(B, V)` where B is batch size and V is vocab size
            prev_scores: scores of previous tokens
              :math:`(B)` where B is batch size

        Returns:
            - log probability of generated sequence
            - generated sequence
        """
        scores = create_init_scores(tokens, memory) if prev_scores is None else prev_scores
        for _ in range(int(memory.size(0) * self._maxlen_a + self._maxlen_b)):
            logits = self._decoder(tokens, memory, memory_padding_mask)
            logits = logits[:, -1, :]
            if target_mask is not None:
                logits = logits.masked_fill(target_mask, float('-inf'))
            next_word_scores, words = self._sample_from(logits)
            eos_mask = words.eq(self._eos)
            scores = scores + next_word_scores.masked_fill_(eos_mask, 0.).view(-1)
            tokens = torch.cat([tokens, words.unsqueeze(dim=-1)], dim=-1)
        return scores, tokens

    def _sample_from(self, logits):
        """
        Sample next word from logits

        Args:
            logits: log probability of next tokens

        Returns:
            - sequence at time step T+1
              :math:`(N, T+1)` where N is batch size, B is beam size and T is current sequence length
            - log probablity of sequence
              :math:`(N)` where N is batch size and B is beam size.
        """
        if self._topk > 0:
            mask = torch.ones_like(logits)
            mask = mask.cumsum(dim=-1) - 1
            mask = mask > self._topk
            logits = logits.masked_fill(mask, float('-inf'))
        prob = F.softmax(logits, dim=-1)
        dist = D.Categorical(prob)
        words = dist.sample()
        scores = logits.gather(-1, words.unsqueeze(dim=-1)).squeeze(dim=-1)
        return scores, words
