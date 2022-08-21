from typing import Dict
import json
import re

import numpy as np

import torch
import torch.nn.functional as F

from paragen.criteria import create_criterion
from paragen.models import create_model
from paragen.tasks import register_task
from paragen.tasks.base_task import BaseTask
from paragen.utils.data import reorganize, possible_eval
from paragen.utils.tensor import convert_idx_to_tensor, convert_tensor_to_idx


@register_task
class GermlineBaseTask(BaseTask):
    """
    Args:
        maxlen: maximum length for sequence
    """

    def __init__(self,
                 requires_tokenize=False,
                 maxlen=1024,
                 sequence_name='sequence',
                 germline_name='germline',
                 label_name = None,
                 num_germline=1,
                 **kwargs,
                 ):
        super().__init__(**kwargs)

        self._maxlen = maxlen
    
        self._sequence_name = sequence_name
        self._germline_name = germline_name
        self._label_name = label_name
        self._string_keys = [sequence_name, germline_name]

        self._num_germline = num_germline
        self._label_padding_idx = -1

        self._split = None
        if requires_tokenize:
            self._split = self.add_space

    def _build_models(self):
        self._model = create_model(self._model_configs)
        self._model.build(vocab_size=len(self._tokenizer),
                          special_tokens=self._tokenizer.special_tokens)

    def _build_criterions(self):
        self._criterion = create_criterion(self._criterion_configs)
        self._criterion.build(self._model, padding_idx=self._label_padding_idx)

    def _data_collate_fn(self, sample, is_training=True) -> Dict:

        processed_sample = {}
        processed_sample['sequence'] = possible_eval(sample[self._sequence_name])
        processed_sample['germline'] = possible_eval(sample[self._germline_name])

        if self._label_name:
            processed_sample['label'] = possible_eval(sample[self._label_name])

        processed_sample['text_sequence'] = processed_sample['sequence']
        processed_sample['text_germline'] = processed_sample['germline']

        if not isinstance(processed_sample['germline'], list):
            processed_sample['germline'] = [processed_sample['germline']]

        if self._split:
            processed_sample['sequence'] =  self._split(processed_sample['sequence'])
            processed_sample['germline'] =  [self._split(germ) for germ in processed_sample['germline']]

        processed_sample['sequence'] = self._tokenizer._bos_token + ' ' + processed_sample['sequence'] + ' ' + self._tokenizer._eos_token
        processed_sample['germline'] = [self._tokenizer._bos_token + ' ' + v + ' ' + self._tokenizer._eos_token for v in processed_sample['germline']]
        processed_sample['sequence'] = self._tokenizer.encode(processed_sample['sequence'])
        processed_sample['germline'] = [self._tokenizer.encode(germ) for germ in processed_sample['germline']]

        # padding germline
        seqlen = len(processed_sample['sequence'])
        processed_sample['germline'] = [v + [0] * (seqlen - len(v)) if len(v) < seqlen else v[:seqlen] for v in processed_sample['germline']]


        return {
            'text': sample,
            'token_num': processed_sample['sequence'].count(" ") + 1,
            'processed': processed_sample
        }


    def _collate(self, samples):
        """
        Create batch from a set of processed samples

        Args:
            a list of samples:

        Returns:
            batch: a processed batch of samples used as neural network inputs
        """
        samples = [sample['processed'] for sample in samples]
        samples = reorganize(samples)
        sequences = convert_idx_to_tensor(samples['sequence'], pad=self._tokenizer.pad)

        # build supervised or self-supervised labels
        label = samples['label'] if 'label' in samples.keys() else None
        label = convert_idx_to_tensor(label, pad=self._label_padding_idx)

        germlines = {}
        for i in range(self._num_germline):
            germline = [t[i] for t in samples['germline']]
            germlines['germline{}'.format(i)] = convert_idx_to_tensor(germline, pad=self._tokenizer.pad)

        text_input = [json.dumps({'sequence': x, 'germline': g}) 
                                    for x, g  in zip(samples['text_sequence'], samples['text_germline'])]
        text_output = label

        if not self._infering:
            batch = {
                'net_input': {
                    'sequence': sequences,
                    'germlines': germlines,
                },
                'net_output': {
                    'target': label
                }
            }
        else:
            batch = {
                'net_input': {
                    'sequence': sequences,
                    'germlines': germlines,
                },
                'net_output': {
                    'target': label
                },
                'text_input': text_input
            }
            batch['text_output'] = text_output

        return batch

    def _output_collate_fn(self, outputs, *args, **kwargs):
        """
        Post process a batch of samples

        Args:
            outputs: a batch of outcome

        Returns:
            sample: a batch of processed sample
        """
        
        sample = args[0]
        target = sample['net_output']['target']

        is_labeling = (target.dim() == 2)
        need_probability = (outputs.dim() == (target.dim() + 1))
        if is_labeling:
            mask = (target != self._label_padding_idx)
            flatten_outputs = outputs[mask]
            if need_probability:
                flatten_outputs = F.softmax(flatten_outputs, -1)
            flatten_outputs = convert_tensor_to_idx(flatten_outputs)

            output_len = mask.sum(-1)        

            head = 0
            processed_outputs = []
            for chunk in output_len:
                output = flatten_outputs[head:head+chunk]
                if need_probability:       
                    processed_outputs.append([x[-1] for x in output])
                else:
                    processed_outputs.append(output)
                head += chunk
        else:
            if need_probability:
                outputs = F.softmax(outputs, -1)
                outputs = convert_tensor_to_idx(outputs)
                processed_outputs = [x[-1] for x in outputs]
            else:
                outputs = convert_tensor_to_idx(outputs)
                processed_outputs = outputs

        return processed_outputs

    def add_space(self, text):
        return ' '.join(list(text))   


@register_task
class AncestorGermlinePredictionTask(GermlineBaseTask):
    """
    Args:
        maxlen: maximum length for sequence
    """

    def __init__(self,
                 requires_tokenize=False,
                 maxlen=1024,
                 sequence_name='sequence',
                 germline_name='germline',
                 num_germline=1,
                 negative_ratio=0.5,
                 **kwargs,
                 ):
        super().__init__(requires_tokenize=requires_tokenize,
                            maxlen=maxlen,
                            sequence_name=sequence_name,
                            germline_name=germline_name,
                            label_name=None,
                            num_germline=num_germline,
                            **kwargs,)
        self._negative_ratio = negative_ratio

    def _collate(self, samples):
        """
        Create batch from a set of processed samples

        Args:
            a list of samples:

        Returns:
            batch: a processed batch of samples used as neural network inputs
        """
        samples = [sample['processed'] for sample in samples]
        samples = reorganize(samples)

        permuted_germline, label, permuted_idx = self.build_contrastive_example(samples['sequence'], samples['germline'])

        sequences = convert_idx_to_tensor(samples['sequence'], pad=self._tokenizer.pad)
        label = convert_idx_to_tensor(label, pad=self._label_padding_idx)

        germlines = {}
        for i in range(self._num_germline):
            germline = [t[i] for t in permuted_germline]
            germline_tokens = convert_idx_to_tensor(germline, pad=self._tokenizer.pad)
            padding_germline_tokens = torch.zeros_like(sequences)
            padding_germline_tokens[:,:germline_tokens.size(-1)] = germline_tokens
            germlines['germline{}'.format(i)] = padding_germline_tokens

        text_input = [json.dumps({'sequence': x, 'germline': g, 'permuted_germline': samples['text_germline'][i]}) 
                                    for x, g, i  in zip(samples['text_sequence'], samples['text_germline'], permuted_idx)]
        text_output = label

        if not self._infering:
            batch = {
                'net_input': {
                    'sequence': sequences,
                    'germlines': germlines,
                },
                'net_output': {
                    'target': label
                }
            }
        else:
            batch = {
                'net_input': {
                    'sequence': sequences,
                    'germlines': germlines,
                },
                'net_output': {
                    'target': label
                },
                'text_input': text_input
            }
            batch['text_output'] = text_output

        return batch
    
    def build_contrastive_example(self, sequences, germlines):
        N = len(sequences)
        mask = np.random.rand(N) < self._negative_ratio
        sample_idx = np.random.randint(0, N, N)

        permuted_germlines, permuted_idx, label = [], [], []
        for i in range(N):
            idx = sample_idx[i] if mask[i] else i
            permuted_germlines.append(germlines[idx])
            permuted_idx.append(idx)
            if germlines[idx] == germlines[i]:
                label.append(1)
            else:
                label.append(0)
        
        return permuted_germlines, label, permuted_idx



@register_task
class MutationPositionPredictionTask(GermlineBaseTask):
    """
    Args:
        maxlen: maximum length for sequence
    """

    def __init__(self,
                 requires_tokenize=False,
                 maxlen=1024,
                 sequence_name='sequence',
                 germline_name='germline',
                 num_germline=1,
                 **kwargs,
                 ):
        super().__init__(requires_tokenize=requires_tokenize,
                            maxlen=maxlen,
                            sequence_name=sequence_name,
                            germline_name=germline_name,
                            label_name=None,
                            num_germline=num_germline,
                            **kwargs,)

        assert num_germline == 1, "germline number must be 1"

    def _collate(self, samples):
        """
        Create batch from a set of processed samples

        Args:
            a list of samples:

        Returns:
            batch: a processed batch of samples used as neural network inputs
        """
        samples = [sample['processed'] for sample in samples]
        samples = reorganize(samples)

        sequences = convert_idx_to_tensor(samples['sequence'], pad=self._tokenizer.pad)

        germlines = {}
        for i in range(self._num_germline):
            germline = [t[i] for t in samples['germline']]
            germlines['germline{}'.format(i)] = convert_idx_to_tensor(germline, pad=self._tokenizer.pad)

        mask = self.build_mutation_mask(sequences, germlines['germline0'])
        label = convert_idx_to_tensor(mask, pad=self._label_padding_idx)
        mutation_token = sequences.masked_fill(1 - label, self._tokenizer.pad)

        mutation_text = [self._tokenizer.decode(output) for output in convert_tensor_to_idx(mutation_token,
                                                                        bos=self._tokenizer.bos,
                                                                        eos=self._tokenizer._length+1,
                                                                        pad=self._tokenizer.pad)]


        text_input = [json.dumps({'sequence': x, 'germline': g}) 
                                    for x, g in zip(samples['text_sequence'], samples['text_germline'])]
        text_output = [json.dumps({'mutation_position': x, 'mutation_token': g}) 
                                    for x, g in zip(mask, mutation_text)]

        if not self._infering:
            batch = {
                'net_input': {
                    'sequence': sequences,
                    'germlines': germlines,
                },
            }
        else:
            batch = {
                'net_input': {
                    'sequence': sequences,
                    'germlines': germlines,
                },
                'net_output': {
                    'position': label,
                    'mutation': mutation_token
                },
                'text_input': text_input
            }
            batch['text_output'] = text_output

        return batch

    def build_mutation_mask(self, sequences, germlines):
        sequences_token_np = sequences.cpu().numpy().copy()
        germlines_token_np = germlines.cpu().numpy().copy()

        mask = np.zeros_like(sequences_token_np)
        mask[sequences_token_np != germlines_token_np] = 1
                
        return mask.tolist()
