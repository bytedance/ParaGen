import random
from typing import Dict

import torch

from paragen.criteria import create_criterion
from paragen.models import create_model
from paragen.tasks import register_task
from paragen.tasks.base_task import BaseTask
from paragen.utils.tensor import convert_idx_to_tensor


@register_task
class MaskedLMTask(BaseTask):
    def __init__(self,
                 src,
                 maxlen=512,
                 mask_ratio=0.15,
                 mask_method="word",
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self._src = src
        self._maxlen = maxlen
        self._mask_ratio = mask_ratio
        self._mask_method = mask_method
        assert mask_method in ("word", "span")

    def _build_models(self):
        self._model = create_model(self._model_configs)
        self._model.build(vocab_size=len(self._tokenizer),
                          special_tokens=self._tokenizer.special_tokens,)

    def _build_criterions(self):
        self._criterion = create_criterion(self._criterion_configs)
        self._criterion.build(self._model, padding_idx=self._tokenizer.pad)

    def _data_collate_fn(self, sample, is_training=True) -> Dict:
        sample = sample.strip()
        return {
            'text': sample,
            'token_num': sample.count(" "),
            'processed': self._tokenizer.encode(sample)
        }

    def _collate(self, samples):
        samples = [[self._tokenizer.bos] + sample['processed'] + [self._tokenizer.eos] for sample in samples]
        # samples = reorganize(samples)
        tokens = convert_idx_to_tensor(samples, pad=self._tokenizer.pad)
        # shape of mask, src_tokens, tgt_tokens: bsz x seqlen
        mask = self.generate_mask(tokens, self._mask_ratio, self._mask_method)
        src_tokens, tgt_tokens = self.apply_mask(tokens, mask, split811=True, vocab_size=len(self._tokenizer))
        batch = {
            'net_input': {
                'src_tokens': src_tokens,
                'mask': mask,
                'tgt_tokens': tgt_tokens
            },
        }
        return batch

    def generate_mask(self, src_tokens, mask_prob, mask_method):
        # mask_prob can be a float or a tensor
        legal_mask = self.get_legal_mask(src_tokens)

        if mask_method == 'word':
            while True:
                mask = torch.rand_like(src_tokens.float()) < mask_prob
                mask = mask * legal_mask
                if mask.sum() != 0:
                    return mask
        if mask_method == 'span':
            bsz, seq_len = src_tokens.shape
            mask = torch.zeros_like(src_tokens).bool()
            nonpad_lens = (src_tokens != self._tokenizer.pad).sum(1)
            # ensure that at least a word is masked!
            mask_lens = (nonpad_lens * mask_prob).int() + 1   
            max_lids = (nonpad_lens - mask_lens).tolist()
            for bid in range(bsz):
                span_lid = random.randint(1, max_lids[bid])
                span_rid = span_lid + mask_lens[bid]
                mask[bid, span_lid: span_rid] = True
            return mask
        raise Exception("This line should never be called")

    def apply_mask(self, tokens, mask, split811=False, vocab_size=-1):
        tgt_tokens = tokens.masked_fill(~mask, self._tokenizer.pad)
        if not split811:
            src_tokens = tokens.masked_fill(mask, self._tokenizer.unk)
        else:
            assert vocab_size != -1
            # leave 10% for unchanged
            mask = mask & (torch.rand(mask.shape, device=mask.device) < 0.9)
            src_tokens = tokens.masked_fill(mask, self._tokenizer.unk)
            # replace 1/9 of [mask] with randomized words
            # so that 80% are [mask] and 10% are randomized words
            mask = mask & (torch.rand(mask.shape, device=mask.device) < 0.111111)
            rand_tokens = torch.randint_like(src_tokens, high=vocab_size)
            src_tokens = src_tokens * ~ mask + rand_tokens * mask
        return src_tokens, tgt_tokens

    def get_legal_mask(self, x):
        """
            Args:   [bos, hello,  mask, world, eos, pad, pad]
            Return: [0,   1,      1,    1,     0,   0,   0]
        """
        return (x != self._tokenizer.bos) * (x != self._tokenizer.pad) * (x != self._tokenizer.eos)

    def is_special_token(self, x)->bool:
        return (x == self._tokenizer.bos) or (x == self._tokenizer.pad) or (x == self._tokenizer.eos)


