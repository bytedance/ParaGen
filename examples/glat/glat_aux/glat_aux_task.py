from typing import Dict, List

import torch
from torch import Tensor

from paragen.tasks import register_task
from paragen.tasks.translation_task import TranslationTask
from paragen.utils.data import reorganize
from paragen.utils.tensor import convert_idx_to_tensor
from paragen.trainers import create_trainer


@register_task
class GLATAuxTranslationTask(TranslationTask):
    """
    Seq2SeqTask defines overall scope on sequence to sequence task.

    Args:
        mode (str): process mode. Options: [train, valid, eval, serve]
        model (dict): model configuration to build a neural model
        data (dict): data configuration to build a dataset for train, valid, eval, serve
        tokenizer (dict): tokenization configuration to build a tokenizer to preprocess
        criterion (dict): criterion configuration to build a criterion to compute objective function for a model
        generator (dict): generator configuration to build a generator to produce results in inference
        trainer (dict): trainer configuration to build a trainer to train a model with criterion and optimizer
        evaluator (dict): evaluator configuration to build a evaluator to evaluate the performance of the model
        post_collate (bool): do collate_fn after sampling
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._infering = False

    def _build_trainer(self):
        """
        Build a trainer to schedule training process
        """
        dataloader = self._build_dataloader('train', mode='train')
        eval_dataloaders = {}
        for name, configs in self._dataloader_configs.items():
            if name != 'train':
                eval_dataloaders[name] = self._build_dataloader(name, mode='valid')
        self._trainer = create_trainer(self._trainer_configs)
        self._trainer.build(model=self._model,
                            src_special_tokens=self._tokenizer[self._src].special_tokens,
                            tgt_special_tokens=self._tokenizer[self._tgt].special_tokens,
                            dataloader=dataloader,
                            criterion=self._criterion,
                            eval_dataloaders=eval_dataloaders,
                            evaluator=self._evaluator,
                            task_callback=self._callback,
                            generator=self._generator)

    def _collate(self, samples: List[Dict]) -> Dict[str, List[Tensor]]:
        """
        Create batch from a set of processed samples

        Args:
            a list of samples:

        Returns:
            batch: a processed batch of samples used as neural network inputs
        """
        samples = [sample['processed'] for sample in samples]
        samples = [sample for sample in samples if len(sample[self._src]) > 0 and len(sample[self._tgt]) > 0]
        samples = reorganize(samples)
        src = samples[self._src]
        if self._training:
            src = [v[:self._maxlen] for v in src]
        src = convert_idx_to_tensor(src, pad=self._tokenizer[self._src].pad)
        if not self._infering:
            tgt = samples[self._tgt]
            if self._training:
                tgt = [v[:self._maxlen] for v in tgt]
            tgt = convert_idx_to_tensor(tgt, pad=self._tokenizer[self._tgt].pad)
            tgt_padding_mask = tgt.eq(self._tokenizer[self._tgt].pad)
            tgt_lengths = torch.min((~tgt_padding_mask).sum(-1), torch.tensor(self._maxlen - 1, dtype=torch.long))
            batch = {
                'net_input': {
                    'src': src,
                    'tgt_padding_mask': tgt_padding_mask,
                },
                'net_output': {
                    'token': {
                        'target': tgt
                    },
                    'length': {
                        'target': tgt_lengths
                    }
                },
                'glancing_target': {
                    'target': tgt,
                    'target_padding_mask': tgt_padding_mask
                }
            }
        else:
            net_input = {'src': src}
            batch = {'net_input': net_input, 'text_input': samples['text_input']}
            if 'text_output' in samples:
                batch['text_output'] = samples['text_output']
        return batch

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
                               **kwargs)

