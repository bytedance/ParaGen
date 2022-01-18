from abc import abstractmethod
from collections import OrderedDict
from typing import Dict, Any, Tuple, List
from bisect import bisect_right
from itertools import accumulate
import logging
logger = logging.getLogger(__name__)

import numpy as np

from paragen.datasets import register_dataset, create_dataset, AbstractDataset
from paragen.utils.io import parse_path

from .multi_parallel_text_dataset import MultiParallelTextDataset
from .streaming_multi_parallel_text_dataset import MultiSourceOneSample


class Sampler(object):
    @abstractmethod
    def sample(self, one_sample: Dict[str, Any]):
        pass


class TemperatureSampler(Sampler):
    def __init__(self,
                 temperature: float,
                 dataset_sizes: Dict[str, int]):
        self._keys, self._probabilities = self._calculate_probability(
            temperature,
            dataset_sizes,
        )
        logger.info(f"Sample probabilities: {list(zip(self._keys, self._probabilities))}")

    @staticmethod
    def _calculate_probability(temperature, dataset_sizes) -> Tuple[List[str], List[float]]:
        sizes = list(dataset_sizes.values())
        total_size = sum(sizes)
        prob = [(size / total_size) ** (1.0 / temperature) for size in sizes]
        prob = [p/sum(prob) for p in prob]
        return list(dataset_sizes.keys()), prob

    def sample(self, one_sample: Dict[str, Any]):
        key = np.random.choice(self._keys, p=self._probabilities)
        return one_sample[key], key


def build_sampler(
    sample_strategy: str,
    sample_strategy_args: Any,
) -> Sampler:
    return eval(sample_strategy)(**sample_strategy_args)


@register_dataset
class MultiSourceDataset(AbstractDataset):
    """
    StreamingMultiSourceDataset is an in-memory dataset composed of many different parallel dataset.

    Args:
        lang_pairs: Multiple imported source-target language pairs
        path：Path to load data
        prefix: File name prefix of imported data
        multi_source_dataset: A dict of dataset configs for different source.
            The key of the dict is the source name.
            For multilingual translation, the key is the direction like "en-ja".
        sample_strategy: A sampler class name. The sampler is used to sample from the different sources.
        sample_strategy_args: The parameters to generate the sampler object.
    """

    def __init__(self, 
                 lang_pairs: str,
                 path: str,
                 prefix: str,
                 sample_strategy: str = None,
                 sample_strategy_args: str = None):
        super().__init__(None)
        self._multi_source_dataset = self._obtain_data_config(lang_pairs, path, prefix)
        self._datasets: OrderedDict[str, MultiParallelTextDataset] = OrderedDict({k: create_dataset(config) for k, config in self._multi_source_dataset.items()})
        self._sample_strategy = sample_strategy
        self._sample_strategy_args = sample_strategy_args
        self._sample_pool = {}

    def _obtain_data_config(self, lang_pairs, paths, prefix):
        multi_source_dataset = {}
        lang_pairs = lang_pairs.split(',')
        paths = paths.split(',')
        files = [[filepath.split('/')[-1] for filepath in parse_path(path)] for path in paths]
        for lang_pair in lang_pairs:
            langs = lang_pair.split('-')
            for i, path in enumerate(paths):
                if f"{prefix}.{langs[0]}-{langs[1]}.{langs[0]}" in files[i] and f"{prefix}.{langs[0]}-{langs[1]}.{langs[0] if '_tgt' in langs[1] else langs[1]}" in files[i]:
                    multi_source_dataset[lang_pair] = ({'class': "MultiParallelTextDataset", 
                                                        'path': {langs[0]: f"{path}/{prefix}.{langs[0]}-{langs[1]}.{langs[0]}",
                                                                    langs[1]: f"{path}/{prefix}.{langs[0]}-{langs[1]}.{langs[0] if '_tgt' in langs[1] else langs[1]}"},
                                                        'mono': '_tgt' in langs[1]})
                    break
                elif f"{prefix}.{langs[1]}-{langs[0]}.{langs[0]}" in files[i] and f"{prefix}.{langs[1]}-{langs[0]}.{langs[0] if '_tgt' in langs[1] else langs[1]}" in files[i]:
                    multi_source_dataset[lang_pair] = ({'class': "MultiParallelTextDataset", 
                                                        'path': {langs[0]: f"{path}/{prefix}.{langs[1]}-{langs[0]}.{langs[0]}",
                                                                    langs[1]: f"{path}/{prefix}.{langs[1]}-{langs[0]}.{langs[0] if '_tgt' in langs[1] else langs[1]}"},
                                                        'mono': '_tgt' in langs[1]})
                    break
            else:
                logger.info(f"Not Found dataset of language pair: {lang_pair}")
                raise IOError
        return multi_source_dataset

                    

    def build(self, *args, **kwargs):
        self._pos = 0
        for dataset in self._datasets.values():
            dataset.build(*args, **kwargs)
        self._accumulate_data_size = list(accumulate([len(dataset) for dataset in self._datasets.values()]))
        if self._sample_strategy is not None:
            self._sample_strategy_args['dataset_sizes'] = {k: len(dataset) for k, dataset in self._datasets.items()}
            self._sampler = build_sampler(self._sample_strategy, self._sample_strategy_args)

    def finalize(self):
        for dataset in self._datasets.values():
            dataset.finalize()

    def shuffle(self):
        for dataset in self._datasets.values():
            dataset.shuffle()

    def _callback(self, sample: MultiSourceOneSample) -> MultiSourceOneSample:
        return MultiSourceOneSample(
            key=sample.key,
            sample=self._datasets[sample.key]._callback(sample.sample)
        )

    def __getitem__(self, index):
        if self._sample_strategy is not None:
            sample_next, key = self._sampler.sample(self._sample_pool)
            self._sample_pool[key] = self._next_in_one_dataset(self._datasets[key])
            return sample_next
        else:
            i = bisect_right(self._accumulate_data_size, index)
            if i != 0:
                index = index - self._accumulate_data_size[i-1]
            return list(self._datasets.values())[i][index]

    def __next__(self):
        self._pos += 1
        if self._pos > len(self):
            self._pos = 0
            raise StopIteration
        if self._sample_strategy is not None:
            sample_next, key = self._sampler.sample(self._sample_pool)
            self._sample_pool[key] = self._next_in_one_dataset(self._datasets[key])
            return sample_next
        else:
            one_next = self._next_in_multi_dataset()
            return one_next
        
    def _next_in_multi_dataset(self):
        for dataset in self._datasets.values():
            try:
                return next(dataset)
            except StopIteration:
                continue
        for dataset in self._datasets.values():
            dataset.reset()
        return self._next_in_multi_dataset()
        
    def _next_in_one_dataset(self, dataset: MultiParallelTextDataset):
        try:
            return next(dataset)
        except StopIteration:
            dataset.reset()
            return self._next_in_one_dataset(dataset)

    def reset(self):
        self._pos = 0
        for dataset in self._datasets.values():
            dataset.reset()
        self._sample_pool = {k: self._next_in_one_dataset(dataset) for k, dataset in self._datasets.items()}

    def __len__(self):
        return sum([len(dataset) for dataset in self._datasets.values()])
