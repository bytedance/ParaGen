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
from paragen.utils.tensor import to_device, possible_autocast, split_samples


@register_evaluator
class Evaluator(AbstractEvaluator):
    """
    A general evaluation scheduler for base task

    Args:
        metric (dict): metric configuration for building evaluator
        display_samples (int): the number of samples with hypothesis and references to display
        no_display_option (str): option ['source', 'reference'], default None. Do not display source or reference of a sample
        save_hypo_dir (str): directory path to store hypothesis. All the hypothesis of each dataloader will be saved
            under `save_hypo_dir`
    """

    def __init__(self,
                 metric: Dict = None,
                 display_samples: int = 5,
                 no_display_option: str = None,
                 save_hypo_dir: str = None,
                 ):
        super().__init__()
        self._display_samples = display_samples
        self._save_hypo_dir = save_hypo_dir
        self._metric_configs = metric
        self._no_display_option = no_display_option.lower().split(',') if no_display_option is not None else []

        self._generator, self._dataloaders, self._tokenizer, self._task_callback = None, None, None, None
        self._metric, self._postprocess = None, None
        self._env = None

        self._random_indices = [random.randint(0, 100000000) for _ in range(self._display_samples)]

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
        metric_names = set()
        for data_name, dataloader in self._dataloaders.items():
            logger.info(f'eval on {data_name} dataset')
            self._eval_reset()
            self.eval_one_dataset(dataloader,
                                  out_path=f'{self._save_hypo_dir}/{data_name}.hypo' if self._save_hypo_dir else None)
            self._eval_update()
            metric_logging = []
            for metric_name, metric in self._metric.items():
                scores = metric.eval()
                if isinstance(scores, float):
                    states[f'{data_name}.{metric_name}'] = scores
                    metric_names.add(metric_name)
                    metric_logging.append((f'{data_name}.{metric_name}', scores))
                elif isinstance(scores, Dict):
                    for k, v in scores.items():
                        states[f'{data_name}.{metric_name}-{k}'] = v
                        metric_names.add(f'{metric_name}-{k}')
                        metric_logging.append((f'{data_name}.{metric_name}-{k}', v))
            if len(metric_logging) > 0:
                logger.info(' | '.join([f'{name}: {scores}' for name, scores in metric_logging]))
        for metric_name in metric_names:
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
            samples = [samples]
            while len(samples) > 0:
                s = samples.pop(0)
                try:
                    hypos = self._step(s)
                    hypos = self._postprocess(hypos, s)
                    input_list, hypo_list, ref_list = self._step_update(input_list,
                                                                        hypo_list,
                                                                        ref_list,
                                                                        s['text_input'] if 'text_input' in s else [None for _ in hypos],
                                                                        hypos,
                                                                        s['text_output'] if 'text_output' in s else [None for _ in hypos])
                except RuntimeError as e:
                    error_info = str(e)
                    error_info = error_info.split('\n')[0]
                    logger.warning(error_info)
                    oom = 'out of memory' in error_info
                    if not oom:
                        raise e
                    if oom and self._env.device == 'cuda':
                        torch.cuda.empty_cache()
                        s1, s2 = split_samples(s)
                        samples.extend([s1, s2])

        info = ''
        for idx in self._random_indices:
            idx = idx % len(hypo_list)
            info += '\n'
            if 'source' not in self._no_display_option and input_list[idx] is not None:
                info += f'\tSource: {input_list[idx]}\n'
            info += f'\tHypothesis: {hypo_list[idx]}\n'
            if 'reference' not in self._no_display_option and ref_list[idx] is not None:
                info += f'\tReference: {ref_list[idx]}\n'
        logger.info(info)
        if self._env.is_master() and out_path:
            with UniIO(out_path, 'w') as fout:
                for hypo in hypo_list:
                    fout.write('{}\n'.format(hypo))
        if self._metric:
            for metric in self._metric.values():
                metric.add_all(hypo_list, ref_list)

    def _step(self, samples):
        self._step_reset()
        with torch.no_grad():
            self._generator.reset(mode='infer')
            samples = to_device(samples, self._env.device)
            if isinstance(samples['net_input'], Dict):
                samples['net_input'] = auto_map_args(samples['net_input'], self._generator.input_slots)
            with possible_autocast():
                hypos = self._generator(*samples['net_input'])
        return hypos

