import torch
from torch import Tensor
from typing import Callable, Tuple
from paragen.utils.ops import local_seed
from paragen.modules.utils import create_padding_mask_from_length
from paragen.modules.search.abstract_search import AbstractSearch

"""
Args: window_size L

input: length [B], encoder_out [S * B * D], encoder_padding_mask [B * S]

mid: l' in [length-window_size, length+window_size] [B, 2*L+1]
predict sequennce candidate for each l' [B, 2 * L + 1, 2 * L + 1], [B, 2 * L + 1]
rerank candidates [B, 2*L+1]

output: sequence
"""
class GLATLengthSearcher(AbstractSearch):
    def __init__(self,
                 window_size=5,
                 max_len=256,
                 seed=None,
                 padding_token=None,
                 calc_decoder_input: Callable[[Tensor, Tensor], Tensor]=None,
                 decoder=None) -> None:
        super().__init__()
        self._window_size = window_size
        self._max_len = max_len
        self._seed = seed
        self._padding_token = padding_token
        self._calc_decoder_input = calc_decoder_input
        self._decoder = decoder

    def build(self, *args, **kwargs):
        pass

    def forward(self, 
               length: Tensor,
               src_padding_mask: Tensor,
               src_hidden: Tensor) -> Tensor:
        _lower_bound = torch.tensor(1).to(length)
        _upper_bound = torch.tensor(self._max_len).to(length)
        maxlen = torch.minimum(_upper_bound, length.max() + self._window_size)
        candidates = list()
        for offset in range(-self._window_size, self._window_size + 1):
            _length = length + offset
            _length = torch.maximum(_lower_bound, _length)
            _length = torch.minimum(_upper_bound, _length)
            candidates.append(self._search(_length, maxlen, src_padding_mask, src_hidden))
        scores = torch.stack([candidate[1] for candidate in candidates])
        best_idxs = scores.max(dim=0).indices
        outputs = torch.stack([candidates[idx][0][i] for i, idx in enumerate(best_idxs)])
        return outputs

    def _search(self,
                length: Tensor,
                maxlen,
                src_padding_mask: Tensor,
                src_hidden: Tensor) -> Tuple[Tensor, Tensor]:
        tgt_padding_mask = create_padding_mask_from_length(length, maxlen)
        decoder_input = self._calc_decoder_input(src_padding_mask, tgt_padding_mask)
        with local_seed(self._seed):
            logits = self._decoder(decoder_input[0],
                                   src_hidden,
                                   tgt_padding_mask,
                                   src_padding_mask)
        prob, decoder_output = logits.max(dim=-1)
        score = torch.sum(prob * (~tgt_padding_mask), dim=-1) / length
        decoder_output = decoder_output.masked_fill_(tgt_padding_mask, self._padding_token)
        return decoder_output, score

