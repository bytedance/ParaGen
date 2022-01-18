from contextlib import contextmanager
from typing import List, Dict
import random

from paragen.criteria import create_criterion
from paragen.models import create_model
from paragen.tasks import register_task
from paragen.tasks.base_task import BaseTask
from paragen.generators import create_generator
from paragen.utils.data import split_tgt_sequence, reorganize, mask_seq, delete_token, infill_text, permute, rotate
from paragen.utils.tensor import convert_idx_to_tensor, convert_tensor_to_idx


@register_task
class AutoEncodingTask(BaseTask):

    def __init__(self,
                 mode,
                 maxlen=512,
                 model=None,
                 data=None,
                 tokenizer=None,
                 criterion=None,
                 generator=None,
                 trainer=None,
                 evaluator=None,
                 preprocessed=False,
                 post_collate=False,
                 masked_input_p=None,
                 delete_input_p=None,
                 infill_input_lam=None,
                 permute_input=False,
                 rotate_input=False,
                 ):
        super().__init__(mode=mode,
                         model=model,
                         data=data,
                         tokenizer=tokenizer,
                         criterion=criterion,
                         generator=generator,
                         trainer=trainer,
                         evaluator=evaluator,
                         preprocessed=preprocessed,
                         post_collate=post_collate,
                         )

        self._maxlen = maxlen

        self._noise_methods = []
        if masked_input_p:
            self._noise_methods.append(lambda x: mask_seq(x, masked_input_p, mask=self._tokenizer.unk))
        if delete_input_p:
            self._noise_methods.append(lambda x: delete_token(x, delete_input_p))
        if infill_input_lam:
            self._noise_methods.append(lambda x: infill_text(x, infill_input_lam, mask=self._tokenizer.unk))
        if permute_input:
            self._noise_methods.append(permute)
        if rotate_input:
            self._noise_methods.append(rotate)

    def _build_criterions(self):
        self._criterion = create_criterion(self._criterion_configs)
        self._criterion.build(model=self._model, padding_idx=self._tokenizer.pad)

    def _build_generator(self):
        """
        Build generator for model in inference
        """
        self._generator = create_generator(self._generator_configs)
        self._generator.build(self._model,
                              bos=self._tokenizer.bos,
                              eos=self._tokenizer.eos)

    def _build_models(self):
        self._model = create_model(self._model_configs)
        self._model.build(vocab_size=len(self._tokenizer),
                          padding_idx=self._tokenizer.pad)

    def _collate_fn_dynamic(self, sample) -> Dict:
        target = sample
        sample = self._tokenizer.encode(sample)
        tgt = [s for s in sample]
        if self._noise_methods:
            idx = random.randint(0, len(self._noise_methods) - 1)
            tgt = self._noise_methods[idx](sample)
        sample = {'input': sample, 'target': tgt if not self._infering else target}
        return sample

    def _batch(self, samples):
        batch_size = len(samples)
        samples = reorganize(samples)
        src = [v + [self._tokenizer.eos] for v in samples['input']]
        src = [v[:self._maxlen] for v in src]
        src = convert_idx_to_tensor(src, pad=self._tokenizer.pad)
        if not self._infering:
            tgt, prev_tokens = split_tgt_sequence(samples['target'],
                                                  bos=self._tokenizer.bos,
                                                  eos=self._tokenizer.eos)
            tgt = [v[:self._maxlen] for v in tgt]
            tgt = convert_idx_to_tensor(tgt, pad=self._tokenizer.pad)
            prev_tokens = convert_idx_to_tensor(prev_tokens, pad=self._tokenizer.pad)
            batch = {'net_input': (src, prev_tokens),
                     'net_output': (tgt,)}
        else:
            _, prev_tokens = split_tgt_sequence([[] for _ in range(batch_size)],
                                                bos=self._tokenizer.bos,
                                                eos=self._tokenizer.eos)
            prev_tokens = convert_idx_to_tensor(prev_tokens, pad=self._tokenizer.pad)
            batch = {
                'net_input': {
                    'encoder': (src,),
                    'decoder': (prev_tokens,),
                },
                'gold': samples['target']
            }
        return batch

    def _debatch(self, idx):
        idx = convert_tensor_to_idx(idx,
                                    bos=self._tokenizer.bos,
                                    eos=self._tokenizer.eos,
                                    pad=self._tokenizer.pad)
        return idx

    def _post_collate_fn(self, x, *args, **kwargs):
        x = self._tokenizer.decode(x)
        return x

    @contextmanager
    def _context_callback(self, mode):
        """
        Context management callback

        Args:
            mode: mode in context
        """
        _mode = self._mode
        self._mode = mode
        self._model.set_mode(self._mode)
        yield
        self._mode = _mode
        self._model.set_mode(self._mode)


