from typing import Dict
import json

from mosestokenizer import MosesDetokenizer, MosesTokenizer

from paragen.criteria import create_criterion
from paragen.evaluators import create_evaluator
from paragen.generators import create_generator
from paragen.models import create_model
from paragen.tasks import register_task
from paragen.tasks.base_task import BaseTask
from paragen.tokenizers import create_tokenizer
from paragen.utils.data import reorganize, split_tgt_sequence
from paragen.utils.tensor import convert_idx_to_tensor, convert_tensor_to_idx


@register_task
class TranslationTask(BaseTask):
    """
    TranslationTask defines overall scope on translation task.

    Args:
        src: source language
        tgt: target language
        requires_moses_tokenize: do moses tokenization on data
        maxlen: max length for input sequences
        share_vocab: share source and target vocabulary
        index_only: only do indexing
        post_detok: perform moses detokenization for target sequence
    """

    def __init__(self,
                 src,
                 tgt,
                 requires_moses_tokenize=False,
                 maxlen=512,
                 share_vocab=True,
                 index_only=False,
                 post_detok=True,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self._src, self._tgt = src, tgt
        self._share_vocab = share_vocab
        self._maxlen = maxlen
        self._index_only = index_only
        self._requires_moses_tokenize = requires_moses_tokenize
        self._post_detok = post_detok

        if self._requires_moses_tokenize:
            self._moses_tokenize = {lang: MosesTokenizer(lang=lang) for lang in [src, tgt]}
        self._moses_detokenize = {lang: MosesDetokenizer(lang=lang) for lang in [src, tgt]}

    def _build_tokenizers(self):
        """
        Build tokenizers for source and target languages
        """
        if self._share_vocab:
            tokenizer = create_tokenizer(self._tokenizer_configs)
            tokenizer.build()
            self._tokenizer = {self._src: tokenizer, self._tgt: tokenizer}
        else:
            self._tokenizer = {}
            for name in [self._src, self._tgt]:
                tokenizer = create_tokenizer(self._tokenizer_configs[name])
                tokenizer.build()
                self._tokenizer[name] = tokenizer

    def _build_models(self):
        """
        Build a sequence-to-sequence model
        """
        self._model = create_model(self._model_configs)
        self._model.build(src_vocab_size=len(self._tokenizer[self._src]),
                          tgt_vocab_size=len(self._tokenizer[self._tgt]),
                          src_special_tokens=self._tokenizer[self._src].special_tokens,
                          tgt_special_tokens=self._tokenizer[self._tgt].special_tokens)

    def _build_criterions(self):
        """
        Build a criterion
        """
        self._criterion = create_criterion(self._criterion_configs)
        self._criterion.build(self._model, padding_idx=self._tokenizer[self._tgt].pad)

    def _build_generator(self):
        """
        Build generator for model in inference
        """
        self._generator = create_generator(self._generator_configs)
        self._generator.build(self._model,
                              src_special_tokens=self._tokenizer[self._src].special_tokens,
                              tgt_special_tokens=self._tokenizer[self._tgt].special_tokens)

    def _build_evaluator(self):
        """
        Build a evaluator to schedule evaluation process
        """
        dataloaders = {}
        for name, configs in self._dataloader_configs.items():
            if name != 'train':
                dataloaders[name] = self._build_dataloader(name, mode='infer')
        self._evaluator = create_evaluator(self._evaluator_configs)
        self._evaluator.build(generator=self._generator,
                              dataloaders=dataloaders,
                              tokenizer=self._tokenizer[self._tgt],
                              task_callback=self._callback,
                              postprocess=self.postprocess)

    def _data_collate_fn(self, sample: Dict, is_training=True) -> Dict:
        """
        Process a sample
        """
        processed_sample = {}
        for key in [self._src, self._tgt]:
            if key in sample:
                val = sample[key]
                if self._requires_moses_tokenize:
                    val = ' '.join(self._moses_tokenize[key](val))
                processed_sample[key] = self._tokenizer[key].encode(val) if not self._index_only else self._tokenizer[key].token2index(val)
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
        sample = {}
        for key in [self._src, self._tgt]:
            if key in textual_sample:
                val = textual_sample[key]
                if self._index_only:
                    val = self._tokenizer[key].detok(val)
                if self._post_detok and not self._requires_moses_tokenize:
                    val = self._moses_detokenize[key](val.split())
                sample[key] = val

        inputs = {key: val for key, val in sample.items() if key != self._tgt}
        if len(inputs) == 1:
            inputs = [v for v in inputs.values()][0]
        else:
            inputs = json.dumps(inputs)
        processed_sample['text_input'] = inputs

        if self._tgt in textual_sample:
            processed_sample['text_output'] = sample[self._tgt]
        return processed_sample

    def _collate(self, samples):
        """
        Transform samples into tensor format

        Args:
            samples: raw samples fetched from dataset

        Returns:
            - neural network input and output
        """
        samples = [sample['processed'] for sample in samples]
        batch_size = len(samples)
        samples = reorganize(samples)
        src = samples[self._src]
        if self._training:
            src = [v[:self._maxlen] for v in src]
        src = convert_idx_to_tensor(src,
                                    pad=self._tokenizer[self._src].pad,
                                    ndim=2)
        if not self._infering:
            tgt, prev_tokens = split_tgt_sequence(samples[self._tgt],
                                                  bos=self._tokenizer[self._tgt].bos,
                                                  eos=self._tokenizer[self._tgt].eos)
            if self._training:
                prev_tokens = [v[:self._maxlen] for v in prev_tokens]
                tgt = [v[:self._maxlen] for v in tgt]
            tgt = convert_idx_to_tensor(tgt,
                                        pad=self._tokenizer[self._tgt].pad,
                                        ndim=2)
            prev_tokens = convert_idx_to_tensor(prev_tokens,
                                                pad=self._tokenizer[self._tgt].pad,
                                                ndim=2)
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
                                                bos=self._tokenizer[self._tgt].bos,
                                                eos=self._tokenizer[self._tgt].eos)
            prev_tokens = convert_idx_to_tensor(prev_tokens,
                                                pad=self._tokenizer[self._tgt].pad,
                                                ndim=2)
            net_input = {
                'encoder': (src,),
                'decoder': (prev_tokens,),
            }
            batch = {'net_input': net_input, 'text_input': samples['text_input']}
            if 'text_output' in samples:
                batch['text_output'] = samples['text_output']
        return batch

    def _output_collate_fn(self, outputs, *args, **kwargs):
        """
        Post process a sample

        Args:
            outputs: an outcome

        Returns:
            outputs: a processed sample
        """
        outputs = convert_tensor_to_idx(outputs,
                                        bos=self._tokenizer[self._tgt].bos,
                                        eos=self._tokenizer[self._tgt].eos,
                                        pad=self._tokenizer[self._tgt].pad)
        processed_outputs = []
        for output in outputs:
            output = self._tokenizer[self._tgt].decode(output)
            if self._post_detok:
                output = self._moses_detokenize[self._tgt](output.split())
            processed_outputs.append(output)
        return processed_outputs

    def export(self, path, **kwargs):
        """
        Export model for service

        Args:
            path: export path
        """
        def _fetch_first_sample():
            """
            Fetch first sample as input for tracing generator

            Returns:
                sample: a batch of sample
            """
            for name in self._dataloader_configs:
                if name != 'train':
                    dataloader = self._build_dataloader(name, mode='infer')
                    for sample in dataloader:
                        return sample['net_input']
        self._infering = True
        self._generator.export(path,
                               _fetch_first_sample(),
                               bos=self._tokenizer[self._tgt].bos,
                               eos=self._tokenizer[self._tgt].eos,
                               **kwargs)

