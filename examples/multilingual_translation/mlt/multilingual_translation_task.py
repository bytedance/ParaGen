from typing import Dict, List
import json
import os

import numpy as np
from tensorflow.io.gfile import GFile
from mosestokenizer import MosesDetokenizer, MosesTokenizer

from paragen.criteria import create_criterion
from paragen.evaluators import create_evaluator
from paragen.generators import create_generator
from paragen.models import create_model
from paragen.tasks import register_task
from paragen.tasks.base_task import BaseTask
from paragen.tokenizers import create_tokenizer
from paragen.utils.data import split_tgt_sequence
from paragen.utils.tensor import convert_idx_to_tensor, convert_tensor_to_idx

from .streaming_multi_parallel_text_dataset import MultiSourceOneSample

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@register_task
class MultilingualTranslationTask(BaseTask):
    """
    MultilingualTranslationTask defines overall scope on multilingual translation task.

    Args:
        langs: language list
        requires_moses_tokenize: do moses tokenization on data
        maxlen: max length for input sequences
        share_vocab: share source and target vocabulary
        index_only: only do indexing
        post_detok: perform moses detokenization for target sequence
        src_langtok: way to process the bos token or language token
        tgt_langtok: way to process the bos token or language token
    """

    def __init__(self,
                 langs,
                 tokenized=False,
                 requires_moses_tokenize=False,
                 maxlen=512,
                 index_only=False,
                 post_detok=True,
                 src_langtok=None,
                 tgt_langtok=None,
                 mono=False,
                 sentence_tag='.',
                 input_masked_token='_',
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self._tokenized = tokenized
        self._langs = [lang.strip() for lang in GFile(langs, 'r')]
        self._maxlen = maxlen
        self._max_position = 1024
        self._index_only = index_only
        self._requires_moses_tokenize = requires_moses_tokenize
        self._post_detok = post_detok
        self._mono = mono
        self._sentence_tag = sentence_tag
        self._input_masked_token = input_masked_token

        if self._requires_moses_tokenize:
            self._moses_tokenize = {lang: MosesTokenizer(lang=lang.split('_')[0]) for lang in self._langs}
        self._moses_detokenize = {lang: MosesDetokenizer(lang=lang.split('_')[0]) for lang in self._langs}

        # None for no langtok; "token" for normal style; "replace_bos_token": replace "bos" with langtok
        self._src_langtok = src_langtok
        self._tgt_langtok = tgt_langtok

    def _build_tokenizers(self):
        """
        Build tokenizers for source and target languages
        """
        self._tokenizer = create_tokenizer(self._tokenizer_configs)
        self._tokenizer.build()
        if '_' not in self._langs[0]:
            self._langtok_ids = {lang: self._tokenizer.token2index(f"<{lang}>")[0] for lang in self._langs}
        else:
            self._langtok_ids = {lang: self._tokenizer.token2index(lang)[0] for lang in self._langs}

        self._sentence_tag = self._tokenizer.token2index(self._sentence_tag)[0]
        self._input_masked_token = self._tokenizer.token2index(self._input_masked_token)[
            -1]  # '.' will be encoded to [6, 5]

    def _build_models(self):
        """
        Build a sequence-to-sequence model
        """
        self._model = create_model(self._model_configs)
        self._model.build(vocab_size=len(self._tokenizer),
                          special_tokens=self._tokenizer.special_tokens)

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
        self._generator.build(self._model)

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
                              tokenizer=self._tokenizer,
                              task_callback=self._callback,
                              postprocess=self.postprocess)

    def noise_function(self, instance: List, lam=3.5, mask_percentage=0.35):
        """
        Use sentence permutation and word-span masking for pretraining model
        """
        new_instance = []
        ins_length = len(instance)
        mask_length = int(ins_length * mask_percentage)
        seg_pos = [i + 1 for i in range(ins_length) if instance[i] == self._sentence_tag]
        if len(seg_pos) == 0 or seg_pos[-1] != ins_length:
            seg_pos.append(ins_length)
        # Randomly permute the order of sentences
        if len(seg_pos) > 1:
            idx = np.random.permutation(len(seg_pos))
            for i in idx:
                if i == 0:
                    new_instance.extend(instance[:seg_pos[i]])
                else:
                    new_instance.extend(instance[seg_pos[i - 1]: seg_pos[i]])
        else:
            new_instance = instance

        # Word-span masking
        positions = np.random.choice(ins_length, mask_length, replace=False)
        lengths = np.random.poisson(lam=lam, size=mask_length)
        masked = 0
        for i, position in enumerate(positions):
            if masked >= mask_length:
                break
            length = lengths[i]
            if length == 0:
                length += 1
            if masked + length > mask_length:
                length = mask_length - masked
            masked += length
            new_instance[position: position + length] = [self._input_masked_token] * length

        return new_instance

    def _data_collate_fn(self, sample: MultiSourceOneSample, is_training=True) -> Dict:
        """
        Process a sample
        """
        direction, sample = sample.key, sample.sample
        processed_sample = {}
        processed_sample['direction'] = direction
        text_sample = {}
        for key in direction.split('-'):
            val = sample[key]
            if self._tokenized:
                val = [eval(x) for x in val.split()]
                if f"{key}_text" in sample.keys():
                    text_sample[key] = sample[f"{key}_text"]
                else:
                    text_sample[key] = ""
                    if not is_training:
                        text_sample[key] = self._tokenizer.decode(val)
                processed_sample[key] = val
            else:
                text_sample[key] = val
                if self._requires_moses_tokenize:
                    val = ' '.join(self._moses_tokenize[key](val))
                val = ' '.join(val.split()[:self._maxlen + 1])
                processed_sample[key] = self._tokenizer.encode(val)[
                                        1:-1] if not self._index_only else self._tokenizer.token2index(
                    val)  # remove huggingface mbart tokenizer language token
            processed_sample[key] = processed_sample[key][
                                    :self._maxlen + 1]  # huggingface position cannot be extended automaticall so we delete the part beyond the length in advance
            if self._mono and '_tgt' not in key:
                processed_sample[key] = self.noise_function(processed_sample[key])
            processed_sample[key] = self._append_language_token(processed_sample[key],
                                                                self._langtok_ids[key.replace("_tgt", "")])
        if not is_training:
            processed_sample = self._fill_text_data(direction, processed_sample, text_sample)
        ntokens = len(processed_sample[direction.split('-')[0]])
        if not self._mono:
            ntokens = max(ntokens, len(processed_sample[direction.split('-')[1]]))
        return {
            'text': sample,
            'token_num': ntokens,
            'processed': processed_sample
        }

    def _append_language_token(self, token_list, language_id):
        if language_id is not None:
            token_list = [language_id] + token_list
        return token_list

    def _fill_text_data(self, direction, processed_sample, textual_sample):
        """
        Fill textual data into processed_samples
        Args:
            key: language direction
            processed_sample: processed samples
            textual_sample: textual samples

        Returns:
            processed samples with textual one
        """
        sample = {}
        src, tgt = direction.split('-')
        for key in [src, tgt]:
            val = textual_sample[key]
            if self._index_only:
                val = self._tokenizer.detok(val)
            if self._post_detok and not self._requires_moses_tokenize:
                val = self._moses_detokenize[key.replace("_tgt", "")](val.split())
            sample[key] = val

        inputs = {key: val for key, val in sample.items() if key != tgt}
        if len(inputs) == 1:
            inputs = [v for v in inputs.values()][0]
        else:
            inputs = json.dumps(inputs)
        processed_sample['text_input'] = inputs

        if tgt in textual_sample:
            processed_sample['text_output'] = sample[tgt]
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
        src, tgt, text_info = self._merge_multisource(samples)
        src = [v + [self._tokenizer.special_tokens['eos']] for v in src]
        if self._training:
            src = [v[:self._maxlen] for v in src]
        else:
            src = [v[:self._max_position] for v in src]
        src = convert_idx_to_tensor(src, pad=self._tokenizer.special_tokens['pad'])
        if not self._infering:
            if self._tgt_langtok == 'replace_bos_token':
                prev_tokens = tgt
                tgt = [v[1:] + [self._tokenizer.special_tokens['eos']] for v in tgt]
            else:
                tgt, prev_tokens = split_tgt_sequence(tgt,
                                                      bos=self._tokenizer.special_tokens['bos'],
                                                      eos=self._tokenizer.special_tokens['eos'])
            if self._training:
                prev_tokens = [v[:self._maxlen] for v in prev_tokens]
                tgt = [v[:self._maxlen] for v in tgt]
            else:
                prev_tokens = [v[:self._max_position] for v in prev_tokens]
                tgt = [v[:self._max_position] for v in tgt]
            tgt = convert_idx_to_tensor(tgt, pad=self._tokenizer.special_tokens['pad'])
            prev_tokens = convert_idx_to_tensor(prev_tokens, pad=self._tokenizer.special_tokens['pad'])

            batch = {
                'net_input': {
                    'src': src,
                    'tgt': prev_tokens,
                },
                'net_output': {
                    'target': tgt
                }}
        else:
            tgt_langtok_id = tgt[0][0]
            net_input = {
                'src': src,
                'tgt_langtok_id': tgt_langtok_id
            }
            batch = {'net_input': net_input, 'text_input': text_info[0]}
            if len(text_info[1]) != 0:
                batch['text_output'] = text_info[1]
        return batch

    def _output_collate_fn(self, outputs, *args, **kwargs):
        """
        Post process a batch of samples
        Args:
            outputs: a batch of outcome

        Returns:
            outputs: a batch of processed sample
        """
        outputs = convert_tensor_to_idx(outputs,
                                        bos=self._tokenizer.bos,
                                        eos=self._tokenizer.eos,
                                        pad=self._tokenizer.pad)
        processed_outputs = []
        for output in outputs:
            if self._tokenizer.eos in output:
                output = output[:output.index(self._tokenizer.eos)]
            output = self._tokenizer.decode(output, skip_special_tokens=True)
            if self._post_detok:
                output = self._moses_detokenize[self._langs[0]](output.split())
            if self._tgt_langtok == "token":
                output = ' '.join(output.split()[1:])
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
                dataloader = self._build_dataloader(name, mode='infer')
                for sample in dataloader:
                    return sample['net_input']

        self._infering = True
        self._generator.export(path,
                               _fetch_first_sample(),
                               bos=self._tokenizer.bos,
                               eos=self._tokenizer.eos,
                               **kwargs)

    def _merge_multisource(self, samples):
        src = []
        tgt = []
        text_info = [[], []]
        for sample in samples:
            src_lang, tgt_lang = sample['direction'].split('-')
            src.append(sample[src_lang])
            tgt.append(sample[tgt_lang])
            if 'text_output' in sample:
                text_info[0].append(sample['text_input'])
                text_info[1].append(sample['text_output'])
        return src, tgt, text_info
