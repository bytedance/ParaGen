from typing import Optional

from torch import Tensor

from paragen.modules.search import AbstractSearch


class SequenceSearch(AbstractSearch):
    """
    SequenceSearch algorithms are used to generate a complete sequence with strategies.
    It usually built from a one-step neural model and fledges the model to a full-step generation.
    """

    def __init__(self):
        super().__init__()

        self._decoder = None
        self._bos, self._eos, self._pad = None, None, None

    def build(self, decoder, bos, eos, pad, *args, **kwargs):
        """
        Build the search algorithm with task instances.

        Args:
            decoder: decoder of neural model.
            bos: begin-of-sentence index
            eos: end-of-sentence index
            pad: pad index
        """
        self._decoder = decoder
        self._bos, self._eos, self._pad = bos, eos, pad

    def forward(self,
                prev_tokens: Tensor,
                memory: Tensor,
                memory_padding_mask: Tensor,
                target_mask: Optional[Tensor] = None,
                prev_scores: Optional[Tensor] = None):
        """
        Decoding full-step sequence

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
        raise NotImplementedError

    def reset(self, mode):
        """
        Reset encoder and switch running mode

        Args:
            mode: running mode in [train, valid, infer]
        """
        self._mode = mode
        self._decoder.reset(mode)

