from typing import Dict, List
import json

from paragen.criteria import create_criterion
from paragen.generators import create_generator
from paragen.models import create_model
from paragen.tasks import register_task
from paragen.tasks.base_task import BaseTask
from paragen.utils.data import reorganize, split_tgt_sequence, count_sample_token
from paragen.utils.tensor import convert_idx_to_tensor, convert_tensor_to_idx


@register_task
class Seq2SeqTask(BaseTask):
    """
    Seq2SeqTask defines overall scope on sequence to sequence task.

    Args:
        src: source key in data dict
        tgt: target key in data dict
        lang: task language
        maxlen: maximum length for sequence
        share_vocab: share source and target vocabulary
        index_only: only do indexing
    """

    def __init__(self,
                 src='src',
                 tgt='tgt',
                 lang='zh',
                 maxlen=(256, 256),
                 share_vocab=False,
                 index_only=False,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self._src, self._tgt = src, tgt
        self._lang = lang
        self._share_vocab = share_vocab
        self._maxlen_src, self._maxlen_tgt = maxlen
        self._index_only = index_only

    def _build_models(self):
        """
        Build a sequence-to-sequence model
        """
        self._model = create_model(self._model_configs)
        self._model.build(src_vocab_size=len(self._tokenizer),
                          tgt_vocab_size=len(self._tokenizer),
                          src_special_tokens=self._tokenizer.special_tokens,
                          tgt_special_tokens=self._tokenizer.special_tokens)

    def _build_criterions(self):
        """
        Build a criterion
        """
        self._criterion = create_criterion(self._criterion_configs)
        self._criterion.build(self._model, padding_idx=self._tokenizer.pad)

    def _build_generator(self):
        """
        Build generator for model in inference
        """
        self._generator = create_generator(self._generator_configs)
        self._generator.build(self._model,
                              src_special_tokens=self._tokenizer.special_tokens,
                              tgt_special_tokens=self._tokenizer.special_tokens)

    def _data_collate_fn(self, sample: Dict, is_training=False) -> Dict:
        processed_sample = {}
        for key in [self._src, self._tgt]:
            val = sample[key]
            processed_sample[key] = self._tokenizer.encode(
                val) if not self._index_only else self._tokenizer.token2index(val)

        if not is_training:
            processed_sample = self._fill_text_data(processed_sample, sample)

        ntokens = len(processed_sample[self._src])
        if self._tgt in processed_sample:
            ntokens = max(ntokens, len(processed_sample[self._tgt]))

        return {
            'text': sample,
            'token_num': ntokens,
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
        inputs = {key: val for key, val in textual_sample.items() if key != self._tgt}
        if len(inputs) == 1:
            inputs = [v for v in inputs.values()][0]
        else:
            inputs = json.dumps(inputs)
        processed_sample['text_input'] = inputs

        if self._tgt in textual_sample:
            processed_sample['text_output'] = textual_sample[self._tgt]
        return processed_sample

    def _collate(self, samples: List[Dict]):
        """
        Create batch from a set of processed samples

        Args:
            a list of samples:

        Returns:
            batch: a processed batch of samples used as neural network inputs
        """
        samples = [sample['processed'] for sample in samples]
        batch_size = len(samples)
        samples = reorganize(samples)
        src = [v + [self._tokenizer.eos] for v in samples[self._src]]
        src = [v[:self._maxlen_src] for v in src]
        src = convert_idx_to_tensor(src, pad=self._tokenizer.pad)
        if not self._infering:
            tgt, prev_tokens = split_tgt_sequence(samples[self._tgt],
                                                  bos=self._tokenizer.bos,
                                                  eos=self._tokenizer.eos)
            prev_tokens = [v[:self._maxlen_tgt] for v in prev_tokens]
            tgt = [v[:self._maxlen_tgt] for v in tgt]
            tgt = convert_idx_to_tensor(tgt, pad=self._tokenizer.pad)
            prev_tokens = convert_idx_to_tensor(prev_tokens, pad=self._tokenizer.pad)
            batch = {
                'net_input': {
                    'src': src,
                    'tgt': prev_tokens,
                },
                'net_output': {
                    'target': tgt
                }}
        else:
            _, prev_tokens = split_tgt_sequence([[] for _ in range(batch_size)],
                                                bos=self._tokenizer.bos,
                                                eos=self._tokenizer.eos)
            prev_tokens = convert_idx_to_tensor(prev_tokens, pad=self._tokenizer.pad)
            net_input = {
                'encoder': (src, ),
                'decoder': (prev_tokens, ),
            }
            batch = {'net_input': net_input}
            if 'text_input' in samples:
                batch['text_input'] = samples['text_input']
            if 'text_output' in samples:
                batch['text_output'] = samples['text_output']
        return batch

    def _output_collate_fn(self, sample, *args, **kwargs):
        """
        Parse decoded results by convert tensor to list

        Returns:
            idx (list): debatched idx
        """
        sample = convert_tensor_to_idx(sample,
                                       bos=self._tokenizer.bos,
                                       eos=self._tokenizer.eos,
                                       pad=self._tokenizer.pad)
        sample = [self._tokenizer.decode(s) for s in sample]
        return sample
