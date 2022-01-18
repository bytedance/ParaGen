from typing import Dict
import random
import torch
import logging
logger = logging.getLogger(__name__)

from paragen.evaluators import AbstractEvaluator, register_evaluator
from paragen.metrics import create_metric
from paragen.utils.ops import auto_map_args
from paragen.utils.runtime import progress_bar, Environment
from paragen.utils.io import UniIO, exists, mkdir, remove_tree
from paragen.utils.tensor import to_device, possible_autocast


@register_evaluator
class Evaluator(AbstractEvaluator):
    """
    A general evaluation scheduler for base task

    Args:
        metric (dict): metric configuration for building evaluator
        display_samples (int): the number of samples with hypothesis and references to display
        save_hypo_dir (str): directory path to store hypothesis. All the hypothesis of each dataloader will be saved
            under `save_hypo_dir`
    """

    def __init__(self,
                 metric: Dict = None,
                 display_samples: int = 5,
                 save_hypo_dir: str = None,
                 ):
        super().__init__()
        self._display_samples = display_samples
        self._save_hypo_dir = save_hypo_dir
        self._metric_configs = metric

        self._generator, self._dataloaders, self._tokenizer, self._task_callback = None, None, None, None
        self._metric, self._postprocess = None, None
        self._env = None

    def build(self, generator, dataloaders, tokenizer, task_callback=None, postprocess=None):
        """
        Build evaluator with given args.

        Args:
            generator (paragen.generators.AbstractGenerator): the inference model to generate hypothesis
            dataloaders (dict[paragen.dataloaders.AbstractDataLoader]): a set of dataloaders to evaluate
            tokenizer (paragen.tokenizers.AbstractTokenizer): a tokenizer
            task_callback: building context in task during for evaluation via a callback function
            postprocess: postprocess pipeline to obtain final hypothesis from predicted results (torch.Tensor)
        """
        self._generator = generator
        self._dataloaders = dataloaders
        self._tokenizer = tokenizer
        self._task_callback = task_callback
        self._postprocess = postprocess

        self._build_metrics()
        self._env = Environment()
        if self._env.is_master() and self._save_hypo_dir:
            if exists(self._save_hypo_dir):
                remove_tree(self._save_hypo_dir)
            mkdir(self._save_hypo_dir)

    def _build_metric(self, configs):
        """
        Build evaluation metric

        Args:
            configs (dict): configuration of one metric

        Returns:
            metric (paragen.metrics.Metric): a metric module for evaluation
        """
        metric = create_metric(configs)
        metric.build(self._tokenizer)
        return metric

    def _build_metrics(self):
        """
        Build all the used metrics from metric_configs
        """
        if self._metric_configs:
            self._metric = {name: self._build_metric(configs)
                            for name, configs in self._metric_configs.items()}
        else:
            self._metric = {}

    def finalize(self):
        """
        Finalize evaluator after finishing evaluation
        """
        for d in self._dataloaders:
            d.finalize()

    def _step_update(self, input_list, hypo_list, ref_list, inputs, hypos, refs):
        """
        Update states by step
        """
        for inp, hypo, ref in zip(inputs, hypos, refs):
            input_list.append(inp)
            hypo_list.append(hypo)
            ref_list.append(ref)
        return input_list, hypo_list, ref_list

    def eval(self):
        """
        Evaluation process
        """
        self._task_callback(training=False, infering=True)
        states = {}
        self._generator.eval()
        self._dataloaders = {
            data_name: dataloader.reset()
            for data_name, dataloader in self._dataloaders.items()
        }
        for data_name, dataloader in self._dataloaders.items():
            logger.info(f'eval on {data_name} dataset')
            self._eval_reset()
            self.eval_one_dataset(dataloader,
                                  out_path='{}/{}.hypo'.format(self._save_hypo_dir,
                                                               data_name) if self._save_hypo_dir else None)
            self._eval_update()
            for metric_name, metric in self._metric.items():
                states[f'{data_name}.{metric_name}'] = metric.eval()
            logger.info(' | '.join(['{}: {}'.format(name, metric.eval())
                                    for name, metric in self._metric.items()]))
        for metric_name in self._metric.keys():
            if len(self._dataloaders) > 0:
                tot, cnt = 0, 0
                for data_name in self._dataloaders.keys():
                    tot, cnt = tot + states[f'{data_name}.{metric_name}'], cnt + 1
                states[metric_name] = tot / cnt
        return states

    def _eval_reset(self):
        """
        Reset states before the overall evaluation process
        """
        if self._metric:
            for metric in self._metric.values():
                metric.reset()

    def eval_one_dataset(self, dataloader, out_path=None):
        """
        Evaluation on one dataset

        Args:
            dataloader (paragen.dataloaders.AbstractDataLoader): dataloader to fetch data
            out_path: path to store hypothesis

        """
        input_list, hypo_list, ref_list = [], [], []
        for samples in progress_bar(dataloader):
            self._step_reset()
            with torch.no_grad():
                self._generator.reset(mode='infer')
                samples = to_device(samples, self._env.device)
                if isinstance(samples['net_input'], Dict):
                    samples['net_input'] = auto_map_args(samples['net_input'], self._generator.input_slots)
                with possible_autocast():
                    hypos = self._generator(*samples['net_input'])

            hypos = self._postprocess(hypos, samples)
            input_list, hypo_list, ref_list = self._step_update(input_list,
                                                                hypo_list,
                                                                ref_list,
                                                                samples['text_input'] if 'text_input' in samples else [None for _ in hypos],
                                                                hypos,
                                                                samples['text_output'] if 'text_output' in samples else [None for _ in hypos])
        info = ''
        for _ in range(self._display_samples):
            idx = random.randint(0, len(hypo_list) - 1)
            info += '\n'
            if input_list[idx] is not None:
                info += f'\tInput: {input_list[idx]}\n'
            info += f'\tHypothesis: {hypo_list[idx]}\n'
            if ref_list[idx] is not None:
                info += f'\tGround Truth: {ref_list[idx]}\n'
        logger.info(info)
        if self._env.is_master() and out_path:
            with UniIO(out_path, 'w') as fout:
                for hypo in hypo_list:
                    fout.write('{}\n'.format(hypo))
        if self._metric:
            for metric in self._metric.values():
                metric.add_all(hypo_list, ref_list)


