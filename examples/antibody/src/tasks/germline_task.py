from typing import Dict
import json
import re

import torch.nn.functional as F

from paragen.criteria import create_criterion
from paragen.models import create_model
from paragen.tasks import register_task
from paragen.tasks.base_task import BaseTask
from paragen.utils.data import reorganize, possible_eval
from paragen.utils.tensor import convert_idx_to_tensor, convert_tensor_to_idx


@register_task
class GermlineTask(BaseTask):
    """
    EncoderExtraction defines the extraction of encoder.

    Args:
        maxlen: maximum length for sequence
    """

    def __init__(self,
                 requires_tokenize=False,
                 maxlen=1024,
                 sequence_name='sequence',
                 germline_name='germline',
                 cdr_name='cdrs',
                 label_name='label',
                 target_label=None,
                 label_padding_idx=-1,
                 num_germline=1,
                 **kwargs,
                 ):
        super().__init__(**kwargs)

        self._maxlen = maxlen
    
        self._sequence_name = sequence_name
        self._germline_name = germline_name
        self._cdr_name = cdr_name
        self._label_name = label_name
        self._string_keys = [sequence_name, germline_name, cdr_name]

        self._num_germline = num_germline
        self._label_padding_idx = label_padding_idx

        self.target_label = target_label
        self._label2idx, self._idx2label = {}, {}

        self._split = None
        if requires_tokenize:
            self._split = self.add_space

    def _build_models(self):
        """
        Build a text classification model
        """
        self._model = create_model(self._model_configs)
        print(self._tokenizer.special_tokens)
        self._model.build(vocab_size=len(self._tokenizer),
                          special_tokens=self._tokenizer.special_tokens)

    def _build_criterions(self):
        self._criterion = create_criterion(self._criterion_configs)
        
        if self._criterion_configs['class'] == 'CrossEntropy':
            self._criterion.build(self._model, padding_idx=self._label_padding_idx)
        else:
            self._criterion.build(self._model)

    def _data_collate_fn(self, sample: Dict, **kwargs) -> Dict:
        """
        Process a sample statically, such as tokenization

        Args:
            sample: a sample

        Returns:
            sample: a processed sample
        """
        processed_sample = {'sequence': '', 'germlines': [], 'label': None}
        sample[self._sequence_name] = possible_eval(sample[self._sequence_name])
        sample[self._label_name] = possible_eval(sample[self._label_name])
        sample[self._germline_name] = possible_eval(sample[self._germline_name])
        cdrs = None
        for key, val in sample.items():
            if key == self._sequence_name:
                val = self._split(val) if self._split else val
                val = val.replace('-', self._tokenizer.pad_token)
                processed_sample['sequence'] = val
            elif key == self._germline_name:
                if not isinstance(val, list):
                    val = [val]
                for v in val:
                    v = self._split(v) if self._split else v
                    v = v.replace('-', self._tokenizer.pad_token)
                    processed_sample['germlines'].append(v)
            elif key == self._label_name:
                if self.target_label:
                    processed_sample['label'] = 1 if val == self.target_label else 0
                    sample[self._label_name] = 1 if val == self.target_label else 0
                else:
                    if val == None:
                        sample[self._label_name] = 'None'
                        val = 'None'
                    if isinstance(val, str):
                        if val not in self._label2idx:
                            self._label2idx[val] = len(self._label2idx)
                            self._idx2label[len(self._idx2label)] = val
                        processed_sample['label'] = self._label2idx[val]
                    else:
                        processed_sample['label'] = val
            elif key == self._cdr_name:
                cdrs = val

        processed_sample['sequence'] = self._tokenizer.bos_token + ' ' + processed_sample['sequence'] + ' ' + self._tokenizer.eos_token
        processed_sample['germlines'] = [self._tokenizer.bos_token + ' ' + v + ' ' + self._tokenizer.eos_token for v in processed_sample['germlines']]
        if cdrs:
            processed_sample['label'] = self.expand_label(processed_sample['sequence'], cdrs, processed_sample['label'])
        processed_sample['sequence'] = self._tokenizer.encode(processed_sample['sequence'])
        processed_sample['germlines'] = [self._tokenizer.encode(v) for v in processed_sample['germlines']]

        # padding germline
        seqlen = len(processed_sample['sequence'])
        processed_sample['germlines'] = [v + [self._tokenizer.pad] * (seqlen - len(v)) if len(v) < seqlen else v[:seqlen] for v in processed_sample['germlines']]

        processed_sample = self._fill_text_data(processed_sample, sample)
        return {
            'text': sample,
            'token_num': processed_sample['sequence'].count(" ") + 1,
            'processed': processed_sample
        }

    def _fill_text_data(self, processed_sample, textual_sample):
        """
        Fill textual data into processed_samples
        Args:
            processed_sample: processed samples
            textual_sample: textual samples

        Returns:
            processed samples with textual one
        """

        inputs = {key: val for key, val in textual_sample.items() if key in self._string_keys}
        if len(inputs) == 1:
            inputs = [v for v in inputs.values()][0]
        else:
            inputs = json.dumps(inputs)
        processed_sample['text_input'] = inputs

        if self._label_name in textual_sample:
            if isinstance(textual_sample[self._label_name], list) and isinstance(textual_sample[self._label_name][0], list):
                processed_sample['text_output'] = sum(textual_sample[self._label_name],[])
            else:
                processed_sample['text_output'] = textual_sample[self._label_name]
        return processed_sample


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
        label = convert_idx_to_tensor(samples['label'], pad=self._label_padding_idx)

        germlines = {}
        for i in range(self._num_germline):
            germline = [t[i] for t in samples['germlines']]
            germlines['germline{}'.format(i)] = convert_idx_to_tensor(germline, pad=self._tokenizer.pad)

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
                'text_input': samples['text_input']
            }
            if 'text_output' in samples:
                batch['text_output'] = samples['text_output']

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
                if len(self._idx2label) != 0:
                    processed_outputs = [self._idx2label[x] if x in self._idx2label else x for x in outputs]
                else:
                    processed_outputs = outputs

        return processed_outputs


    def expand_label(self, sequence, cdrs, raw_label):
        if isinstance(cdrs, list):
            cdrs = '|'.join(cdrs)
        else:
            cdrs = cdrs
        cdrs = self._split(cdrs) if self._split else cdrs
        sequence_token = sequence.split()
        clean_sequence = sequence.replace(self._tokenizer.pad_token + ' ', '')
        segs = re.split('({})'.format(cdrs), clean_sequence)
        labels = [[self._label_padding_idx]*len(seq.split()) for seq in segs]
        if not isinstance(raw_label[0], list):
            raw_label = [raw_label]
        for i in range(len(raw_label)):
            labels[i*2+1] = raw_label[i]
        flatten_labels = sum(labels,[])

        # insert 0 for '-'
        for i in range(len(sequence_token)):
            if sequence_token[i] == self._tokenizer.pad_token:
                if i == 0 or i == len(sequence_token)-1:
                    flatten_labels.insert(i, -1)
                elif (i-1>0 and flatten_labels[i-1]==self._label_padding_idx) and \
                    (i+1<len(sequence_token) and flatten_labels[i+1]==self._label_padding_idx):
                    flatten_labels.insert(i, -1)
                else:
                    flatten_labels.insert(i, 0)
        return flatten_labels


    def add_space(self, text):
        return ' '.join(list(text))   

