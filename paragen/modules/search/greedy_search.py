from typing import Optional
import torch

from paragen.modules.search import register_search
from paragen.modules.search.sequence_search import SequenceSearch
from paragen.modules.utils import create_init_scores


@register_search
class GreedySearch(SequenceSearch):
    """
    GreedySearch is greedy search on sequence generation.

    Args:
        maxlen_coef (a, b): maxlen computation coefficient.
            The max length is computed as `(S * a + b)`, where S is source sequence length.
    """

    def __init__(
        self, 
        minlen_coef=(0., 0.),
        maxlen_coef=(1.2, 10)
    ):
        super().__init__()

        self._minlen_a, self._minlen_b = minlen_coef
        self._maxlen_a, self._maxlen_b = maxlen_coef

    def forward(self,
                prev_tokens,
                memory,
                memory_padding_mask,
                target_mask: Optional[torch.Tensor] = None,
                prev_scores: Optional[torch.Tensor] = None):
        """
        Decoding full-step sequence with greedy search

        Args:
            prev_tokens: previous tokens or prefix of sequence
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
        batch_size = prev_tokens.size(0)
        scores = create_init_scores(prev_tokens, memory) if prev_scores is None else prev_scores
        minlen = int(memory.size(0) * self._minlen_a + self._minlen_b)
        maxlen = int(memory.size(0) * self._maxlen_a + self._maxlen_b)
        for i in range(maxlen):
            logits = self._decoder(prev_tokens, memory, memory_padding_mask)
            logits = logits[:, -1, :]
            if i < minlen:
                logits[:, self._eos] = float('-inf')
            if target_mask is not None:
                logits = logits.masked_fill(target_mask, float('-inf'))
            next_word_scores, words = logits.max(dim=-1)
            eos_mask = words.eq(self._eos)
            if eos_mask.long().sum() == batch_size:
                break
            scores = scores + next_word_scores.masked_fill_(eos_mask, 0.).view(-1)
            prev_tokens = torch.cat([prev_tokens, words.unsqueeze(dim=-1)], dim=-1)
        return scores, prev_tokens
