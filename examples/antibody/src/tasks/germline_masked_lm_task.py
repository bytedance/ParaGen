import random
import json
from typing import Dict

import torch

from paragen.criteria import create_criterion
from paragen.models import create_model
from paragen.tasks import register_task
from paragen.tasks.base_task import BaseTask
from paragen.utils.tensor import convert_idx_to_tensor, convert_tensor_to_idx
from paragen.utils.data import reorganize, possible_eval
from paragen.utils.pretrain_utils import generate_span_mask




@register_task
class AnitbodyMaskedLMTask(BaseTask):
    def __init__(self,
                 requires_tokenize=False,
                 maxlen=512,
                 src_name='sequence',
                 trg_name='germline',
                 padding_idx=0,
                 mask_ratio=0.15,
                 mask_method="word",
                 splitratio=[0.9, 0.11111],
                 possion_lambda=2,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self._maxlen = maxlen
    
        self._src_name = src_name
        self._trg_name = trg_name
        self._padding_idx = padding_idx

        self._mask_ratio = mask_ratio
        self._mask_method = mask_method
        self._splitratio = splitratio
        self._possion_lambda = possion_lambda
        assert mask_method in ("word", "span")

        self._split = None
        if requires_tokenize:
            self._split = self.add_space

    def _build_models(self):
        self._model = create_model(self._model_configs)
        self._model.build(vocab_size=len(self._tokenizer),
                          special_tokens=self._tokenizer.special_tokens)

    def _build_criterions(self):
        self._criterion = create_criterion(self._criterion_configs)
        self._criterion.build(self._model, padding_idx=self._tokenizer.pad)

    def postprocess(self, hypos, samples, *args, **kwargs):
        mask = samples['net_input'][1]
        hypos = self._output_collate_fn(hypos, *args, **kwargs)
        hypos_list = hypos.split()
        outputs = []
        head = 0
        for m in mask:
            l = m.sum()
            outputs.append(' '.join(hypos_list[head: head+l]))
            head += l 
        return outputs

    def _output_collate_fn(self, sample, *args, **kwargs):
        """
        Post process a sample

        Args:
            sample: an outcome

        Returns:
            sample: a processed sample
        """
        outputs = convert_tensor_to_idx(sample)
        processed_outputs = self._tokenizer.decode(outputs)
        return processed_outputs

    def _data_collate_fn(self, sample, is_training=True) -> Dict:

        processed_sample = {}
        processed_sample['src'] = possible_eval(sample[self._src_name])
        processed_sample['tgt'] = possible_eval(sample[self._trg_name])

        if self._split:
            processed_sample['src'] =  self._split(processed_sample['src'])
            processed_sample['tgt'] =  self._split(processed_sample['tgt'])

        processed_sample['src'] = self._tokenizer.encode(processed_sample['src'])
        processed_sample['tgt'] = self._tokenizer.encode(processed_sample['tgt'])

        processed = processed_sample['src'] + [self._tokenizer.eos] + processed_sample['tgt']
        processed = processed[:self._maxlen-2]

        return {
            'text': sample,
            'token_num': len(processed),
            'processed': processed
        }

    def _collate(self, samples):
        samples = [[self._tokenizer.bos] + sample['processed'] + [self._tokenizer.eos] for sample in samples]
        tokens = convert_idx_to_tensor(samples, pad=self._tokenizer.pad)
        # shape of mask, src_tokens, tgt_tokens: bsz x seqlen
        mask = self.generate_mask(tokens, self._mask_ratio, self._mask_method)
        src_tokens, tgt_tokens = self.apply_mask(tokens, mask, splitratio=self._splitratio, vocab_size=len(self._tokenizer))
        
        masked_text = [self._tokenizer.decode(output) for output in convert_tensor_to_idx(src_tokens,
                                                                        bos=self._tokenizer.bos,
                                                                        eos=self._tokenizer._length+1,
                                                                        pad=self._tokenizer.pad)]
        ori_text = [self._tokenizer.decode(output) for output in convert_tensor_to_idx(tokens,
                                                                        bos=self._tokenizer.bos,
                                                                        eos=self._tokenizer._length+1,
                                                                        pad=self._tokenizer.pad)]
        text_input = [json.dumps({'original': ori, 'masked': masked}) for ori, masked in zip(ori_text, masked_text)]

        text_output = [self._tokenizer.decode(output) for output in convert_tensor_to_idx(tgt_tokens,
                                                                        bos=self._tokenizer.bos,
                                                                        eos=self._tokenizer._length+1,
                                                                        pad=self._tokenizer.pad)]

        if not self._infering:
            batch = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'mask': mask,
                    'tgt_tokens': tgt_tokens
                },
            }
        else:
            batch = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'mask': mask,
                },
                'net_output': {
                    'target': tgt_tokens
                },
                'text_input': text_input, 
                'text_output': text_output}
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
            mask = generate_span_mask(src_tokens, mask_prob, possion_lambda=self._possion_lambda, cutoff=(1, 10))
            mask = mask * legal_mask
            return mask
        raise Exception("This line should never be called")

    def apply_mask(self, tokens, mask, splitratio=None, vocab_size=-1):
        tgt_tokens = tokens.masked_fill(~mask, self._tokenizer.pad)
        if not splitratio:
            src_tokens = tokens.masked_fill(mask, self._tokenizer.unk)
        else:
            assert vocab_size != -1
            # leave 10% for unchanged
            mask = mask & (torch.rand(mask.shape, device=mask.device) < splitratio[0])
            src_tokens = tokens.masked_fill(mask, self._tokenizer.unk)
            # replace 1/9 of [mask] with randomized words
            # so that 80% are [mask] and 10% are randomized words
            mask = mask & (torch.rand(mask.shape, device=mask.device) < splitratio[1])
            rand_tokens = torch.randint_like(src_tokens, high=vocab_size, low=self._tokenizer.unk)
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

    def add_space(self, text):
        return ' '.join(list(text))

@register_task
class EvolutionMaskedLMTask(AnitbodyMaskedLMTask):
    def __init__(self,
                 requires_tokenize=False,
                 maxlen=512,
                 sequence_name='sequence',
                 germline_name='germline',
                 num_germline=1,
                 padding_idx=0,
                 mask_ratio=0.15,
                 mask_method="word",
                 mask_align=False,
                 splitratio=[0.9, 0.11111],
                 possion_lambda=2,
                 **kwargs
                 ):
        super().__init__(requires_tokenize=requires_tokenize, 
                            maxlen=maxlen, 
                            src_name=sequence_name, 
                            trg_name=germline_name, 
                            padding_idx=padding_idx, 
                            mask_ratio=mask_ratio, 
                            mask_method=mask_method, 
                            splitratio=splitratio, 
                            possion_lambda=possion_lambda, 
                            **kwargs)
        self._mask_align = mask_align
        
        self._num_germline = num_germline
        self._num_seq = num_germline + 1

    def _data_collate_fn(self, sample, is_training=True) -> Dict:

        processed_sample = {}
        processed_sample['sequence'] = possible_eval(sample[self._src_name])
        processed_sample['germline'] = possible_eval(sample[self._trg_name])

        if not isinstance(processed_sample['germline'], list):
            processed_sample['germline'] = [processed_sample['germline']]

        if self._split:
            processed_sample['sequence'] =  self._split(processed_sample['sequence'])
            processed_sample['germline'] =  [self._split(germ) for germ in processed_sample['germline']]

        processed_sample['sequence'] = self._tokenizer.encode(processed_sample['sequence'])
        processed_sample['germline'] = [self._tokenizer.encode(germ) for germ in processed_sample['germline']]

        processed = [processed_sample['sequence'][:self._maxlen-2]] + [germ[:self._maxlen-2] for germ in processed_sample['germline']]
        # processed = [germ[:self._maxlen-2] for germ in processed_sample['germline']] + [processed_sample['sequence'][:self._maxlen-2]]

        return {
            'text': sample,
            'token_num': len(processed),
            'processed': processed
        }

    def _collate(self, samples):
        batch_size = len(samples)
        samples = [[[self._tokenizer.bos] + ins + [self._tokenizer.eos] for ins in sample['processed'][:self._num_seq]] for sample in samples]
        samples = sum(samples, []) # bz * (1+G)
        tokens = convert_idx_to_tensor(samples, pad=self._tokenizer.pad)
        # shape of mask, src_tokens, tgt_tokens: bsz x seqlen
        mask = self.generate_mask(tokens, self._mask_ratio, self._mask_method)
        
        if self._mask_align:
            mask = self.align_mask(mask)

        src_tokens, tgt_tokens = self.apply_mask(tokens, mask, splitratio=self._splitratio, vocab_size=len(self._tokenizer))
        
        masked_text = [self._tokenizer.decode(output) for output in convert_tensor_to_idx(src_tokens,
                                                                        bos=self._tokenizer.bos,
                                                                        eos=self._tokenizer._length+1,
                                                                        pad=self._tokenizer.pad)]
        ori_text = [self._tokenizer.decode(output) for output in convert_tensor_to_idx(tokens,
                                                                        bos=self._tokenizer.bos,
                                                                        eos=self._tokenizer._length+1,
                                                                        pad=self._tokenizer.pad)]

        masked_cat_text = [' '.join(masked_text[i:i+self._num_seq]) for i in range(0, len(masked_text), self._num_seq)]
        ori_cat_text = [' '.join(ori_text[i:i+self._num_seq]) for i in range(0, len(ori_text), self._num_seq)]
        
        text_cat_input = [json.dumps({'original': ori, 'masked': masked}) for ori, masked in zip(ori_cat_text, masked_cat_text)]

        text_output = [self._tokenizer.decode(output) for output in convert_tensor_to_idx(tgt_tokens,
                                                                        bos=self._tokenizer.bos,
                                                                        eos=self._tokenizer._length+1,
                                                                        pad=self._tokenizer.pad)]
        text_cat_output = [' '.join(text_output[i:i+self._num_seq]) for i in range(0, len(text_output), self._num_seq)]

        src_tokens = src_tokens.view(batch_size, self._num_seq, -1)
        mask = mask.view(batch_size, -1)
        tgt_tokens = tgt_tokens.view(batch_size, -1)

        if not self._infering:
            batch = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'mask': mask,
                    'tgt_tokens': tgt_tokens
                },
            }
        else:
            batch = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'mask': mask,
                },
                'net_output': {
                    'target': tgt_tokens
                },
                'text_input': text_cat_input, 
                'text_output': text_cat_output}
        return batch

    def align_mask(self, mask):
        '''
            mask: (B * G, S), B is the batch size, G is the sequence number, and S is the sequence length
        '''
        seq_len = mask.size(-1)
        mask = mask.view(-1, self._num_seq, seq_len)
        mask[:,1:,:] = mask[:,0,:].unsqueeze(1).expand_as(mask[:,1:,:])
        mask = mask.view(-1, seq_len)
        
        return mask