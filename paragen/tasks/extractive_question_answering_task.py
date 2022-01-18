from typing import Dict

from paragen.criteria import create_criterion
from paragen.generators import create_generator
from paragen.models import create_model
from paragen.tasks import register_task
from paragen.tasks.base_task import BaseTask
from paragen.utils.data import reorganize
from paragen.utils.tensor import convert_idx_to_tensor, convert_tensor_to_idx


@register_task
class ExtractiveQuestionAnsweringTask(BaseTask):
    """
    QuestionAnsweringTask defines overall scope on sequence to sequence task.

    Args:
        has_answerable: has answerable question
        discard_mismatched: discard mismatched samples
        context: context key in data dict
        question: question key in data dict
        answer: answer key in data dict
        maxlen: maximum length for encoding sequence
        index_only: do indexing only
    """

    def __init__(self,
                 has_answerable=False,
                 discard_mismatched=False,
                 context='context',
                 question='question',
                 answer='answers',
                 maxlen=512,
                 index_only=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self._has_answerable = has_answerable
        self._discard_mismatched = discard_mismatched
        self._context, self._question, self._answer = context, question, answer
        self._maxlen = maxlen
        self._index_only = index_only

    def _make_collator(self, mode):
        """
        create collator for batch creation

        Args:
            mode: running mode [`train`, `valid`, `infer`]

        Returns:
            - batch collator
        """
        return ExtractiveQuestionAnsweringCollator(special_tokens=self._tokenizer.special_tokens,
                                                   answer=self._answer,
                                                   maxlen=self._maxlen,
                                                   infering=(mode == 'infer'))

    def _build_models(self):
        """
        Build a sequence-to-sequence model
        """
        self._model = create_model(self._model_configs)
        self._model.build(vocab_size=len(self._tokenizer),
                          special_tokens=self._tokenizer.special_tokens,)

    def _build_criterions(self):
        """
        Build a criterion
        """
        self._criterion = create_criterion(self._criterion_configs)
        self._criterion.build(self._model,
                              padding_idx=self._tokenizer.pad)

    def _build_generator(self):
        """
        Build generator for model in inference
        """
        self._generator = create_generator(self._generator_configs)
        self._generator.build(self._model,
                              pad=self._tokenizer.pad)

    def _data_collate_fn(self, sample: Dict, is_training=False) -> Dict:
        """
        Process a sample statically, such as tokenization

        Args:
            sample: a sample

        Returns:
            sample: a processed sample
        """
        def build_token_char_map(encoding):
            char2token, token2char_start, token2char_end = {}, {}, {}
            offset = 0
            for i in range(len(encoding.encodings[0].tokens)):
                try:
                    char_span = encoding.token_to_chars(i)
                    s, e = char_span.start, char_span.end
                    offset = e
                    token2char_start[i], token2char_end[i] = s, e
                    for j in range(s, e):
                        char2token[j] = i
                except TypeError:
                    token2char_start[i] = offset
                    pass
            return char2token, token2char_start, token2char_end

        question, context = sample[self._question], sample[self._context]

        kwargs = {'truncation': True, 'max_length': self._maxlen} if is_training else {}
        encoding = self._tokenizer(question,  context, **kwargs)
        char2token, token2char_start, token2char_end = build_token_char_map(encoding)
        processed_sample = {
            'raw_context': sample[self._context],
            'question': sample[self._question],
            'input': encoding['input_ids'],
            'char2token': char2token,
            'token2char_start': token2char_start,
            'token2char_end': token2char_end,
            'text_input': f'{sample[self._question]} [SEP] {sample[self._context]}',
            'token_num': 1
        }
        if self._answer in sample:
            answer = sample[self._answer]
            if answer:
                char_start, char_end = answer['answer_start'], answer['answer_end']
                token_start = char2token[char_start]
                token_end = char2token[char_end-1]
                answerable = True
            else:
                token_start, token_end = 0, 0
                answerable = False
            processed_sample['start'], processed_sample['end'] = token_start, token_end
            processed_sample['answerable'] = answerable
            processed_sample[self._answer] = answer['text']

            if self._discard_mismatched and is_training:
                # do some data cleaning, discard samples unable to reconstruct
                predict = context[token2char_start[token_start]:token2char_end[token_end]]
                gold = answer['text']
                assert predict == gold, f'prediction {predict} mismatch with gold {gold}'
        return processed_sample

    def _collate(self, samples):
        """
        Create batch from a set of processed samples

        Args:
            a list of samples:

        Returns:
            batch: a processed batch of samples used as neural network inputs
        """
        samples = reorganize(samples)
        inp = [v[:self._maxlen] for v in samples['input']]
        inp = convert_idx_to_tensor(inp, pad=self._tokenizer.pad)
        if not self._infering:
            if samples['answerable']:
                start_pos = convert_idx_to_tensor(samples['start'], pad=0)
                end_pos = convert_idx_to_tensor(samples['end'], pad=0)
            else:
                start_pos, end_pos = None, None
            answerable = convert_idx_to_tensor(samples['answerable'], pad=False)
            batch = {
                'net_input': {'input': inp,
                              'answerable': answerable,
                              'start_positions': start_pos,
                              'end_positions': end_pos},
            }
        else:
            net_input = {'input': inp}
            batch = {
                'net_input': net_input,
                'text_input': samples['text_input']
            }
            if self._answer in samples:
                for key in ['raw_context', 'token2char_start', 'token2char_end']:
                    batch[key] = samples[key]
                batch['text_output'] = samples[self._answer]
        return batch

    def _output_collate_fn(self, outputs, samples=None, *args, **kwargs):
        """
        Post process a sample

        Args:
            sample: an outcome

        Returns:
            sample: a processed sample
        """
        outputs = tuple([convert_tensor_to_idx(o) for o in outputs])
        outputs = list(zip(*outputs))

        processed_output = []
        for output, context, token2char_start, token2char_end in zip(outputs,
                                                                     samples['raw_context'],
                                                                     samples['token2char_start'],
                                                                     samples['token2char_end']):
            if self._has_answerable:
                a, s, e = output
                sample = context[token2char_start[s]:token2char_end[e]] if a else self._tokenizer.pad
            else:
                s, e = output
                sample = context[token2char_start[s]:token2char_end[e]]
            processed_output.append(sample)
        return processed_output
