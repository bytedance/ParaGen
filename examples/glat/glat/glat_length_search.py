import torch
from torch import Tensor
from typing import Callable, Tuple
from paragen.utils.ops import local_seed
from paragen.modules.utils import create_padding_mask_from_length
from paragen.modules.search import AbstractSearch, register_search


@register_search
class GLATLengthSearcher(AbstractSearch):
    """
    Args: window_size L

    input: length [B], encoder_out [S * B * D], encoder_padding_mask [B * S]

    mid: l' in [length-window_size, length+window_size] [B, 2*L+1]
    predict sequennce candidate for each l' [B, 2 * L + 1, 2 * L + 1], [B, 2 * L + 1]
    rerank candidates [B, 2*L+1]

    output: sequence
    """
    def __init__(self,
                 beam=7,
                 ) -> None:
        super().__init__()
        self._beam = beam

        self._maxlen = None
        self._seed = None
        self._padding_token = None
        self._calc_decoder_input = None
        self._decoder = None

    def build(self,
              maxlen=256,
              seed=None,
              padding_token=None,
              calc_decoder_input: Callable[[Tensor, Tensor], Tensor] = None,
              decoder=None):
        self._maxlen = maxlen
        self._seed = seed
        self._padding_token = padding_token
        self._calc_decoder_input = calc_decoder_input
        self._decoder = decoder

    def forward(self,
                length: Tensor,
                src: Tensor,
                src_padding_mask: Tensor,
                src_hidden: Tensor) -> Tensor:
        bsz, srclen = src_padding_mask.shape
        window = torch.arange(self._beam, device=length.device) - (self._beam // 2)
        length = length[:, None] + window[None, :]
        length = length.clamp(2, self._maxlen)
        length = length.reshape(-1)
        maxlen = length.max()
        src = src[:, None].repeat(1, self._beam, 1).reshape(bsz * self._beam, -1)
        src_padding_mask = src_padding_mask[:, None].repeat(1, self._beam, 1).reshape(bsz * self._beam, -1)
        src_hidden = src_hidden[:, :, None].repeat(1, 1, self._beam, 1).reshape(srclen, bsz * self._beam, -1)
        candidates, scores = self._search(length, maxlen, src, src_padding_mask, src_hidden)
        candidates = candidates.reshape(bsz, self._beam, -1)
        scores = scores.reshape(bsz, self._beam)
        best_idx = scores.max(dim=1).indices
        outputs = candidates.gather(1, best_idx[:, None, None].repeat(1, 1, maxlen)).squeeze(1)
        return outputs

    def _search(self,
                length: Tensor,
                maxlen,
                src: Tensor,
                src_padding_mask: Tensor,
                src_hidden: Tensor) -> Tuple[Tensor, Tensor]:
        tgt_padding_mask = create_padding_mask_from_length(length, maxlen)
        decoder_input = self._calc_decoder_input(src_padding_mask, tgt_padding_mask, source=src)
        with local_seed(self._seed):
            logits = self._decoder(decoder_input,
                                   src_hidden,
                                   tgt_padding_mask,
                                   src_padding_mask)
        prob, decoder_output = logits.max(dim=-1)
        score = torch.sum(prob * (~tgt_padding_mask), dim=-1) / length
        decoder_output = decoder_output.masked_fill_(tgt_padding_mask, self._padding_token)
        return decoder_output, score

