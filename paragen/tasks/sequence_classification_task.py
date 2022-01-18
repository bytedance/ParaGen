from typing import Dict
import json

from mosestokenizer import MosesTokenizer

from paragen.criteria import create_criterion
from paragen.models import create_model
from paragen.tasks import register_task
from paragen.tasks.base_task import BaseTask
from paragen.utils.data import reorganize, possible_eval
from paragen.utils.tensor import convert_idx_to_tensor, convert_tensor_to_idx


@register_task
class SequenceClassificationTask(BaseTask):
    """
    SequenceClassificationTask defines overall scope on text classification task.

    Args:
        label_name: label key in data dict
        maxlen: maximum length for sequence
    """

    def __init__(self,
                 lang=None,
                 requires_moses_tokenize=False,
                 label_name='label',
                 maxlen=1024,
                 **kwargs,
                 ):
        super().__init__(**kwargs)

        self._label_name = label_name
        self._maxlen = maxlen
        self._label2idx, self._idx2label = {}, {}

        if requires_moses_tokenize:
            assert lang is not None, 'lang should be specified once requires moses tokenize'
            self._moses_tokenize = MosesTokenizer(lang=lang)
            self._moses = lambda sent: ' '.join(self._moses_tokenize(sent))
        else:
            self._moses = None

    def _build_models(self):
        """
        Build a text classification model
        """
        self._model = create_model(self._model_configs)
        self._model.build(vocab_size=len(self._tokenizer),
                          special_tokens=self._tokenizer.special_tokens)

    def _build_criterions(self):
        """
        Build a criterion
        """
        self._criterion = create_criterion(self._criterion_configs)
        self._criterion.build(self._model)

    def _data_collate_fn(self, sample: Dict, **kwargs) -> Dict:
        """
        Process a sample statically, such as tokenization

        Args:
            sample: a sample

        Returns:
            sample: a processed sample
        """
        processed_sample = {'sequence': [], 'label': None}
        sample[self._label_name] = possible_eval(sample[self._label_name])
        for key, val in sample.items():
            if key != self._label_name:
                if self._moses is not None:
                    val = self._moses(val)
                processed_sample['sequence'].append(val)
            else:
                if val not in self._label2idx:
                    self._label2idx[val] = len(self._label2idx)
                    self._idx2label[len(self._idx2label)] = val
                processed_sample['label'] = self._label2idx[val]
        processed_sample['sequence'] = self._tokenizer.encode(processed_sample['sequence'])[:self._maxlen]
        processed_sample = self._fill_text_data(processed_sample, sample)
        return {
            'text': sample,
            'token_num': 1,
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

        inputs = {key: val for key, val in textual_sample.items() if key != self._label_name}
        if len(inputs) == 1:
            inputs = [v for v in inputs.values()][0]
        else:
            inputs = json.dumps(inputs)
        processed_sample['text_input'] = inputs

        if self._label_name in textual_sample:
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
        texts = convert_idx_to_tensor(samples['sequence'], pad=self._tokenizer.pad)
        if not self._infering:
            label = convert_idx_to_tensor(samples['label'], self._tokenizer.pad)
            batch = {
                'net_input': {
                    'input': texts
                },
                'net_output': {
                    'target': label
                }
            }
        else:
            batch = {
                'net_input': {
                    'input': texts,
                },
                'text_input': samples['text_input']}
            if 'text_output' in samples:
                batch['text_output'] = samples['text_output']
        return batch

    def _output_collate_fn(self, outputs, *args, **kwargs):
        """
        Post process a sample

        Args:
            sample: an outcome

        Returns:
            sample: a processed sample
        """
        outputs = convert_tensor_to_idx(outputs)
        processed_outputs = []
        for output in outputs:
            processed_outputs.append(self._idx2label[output])
        return processed_outputs

