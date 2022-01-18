from typing import Dict
import logging
logger = logging.getLogger(__name__)

from tensorflow.io.gfile import GFile

from paragen.datasets import register_dataset
from paragen.utils.runtime import Environment, progress_bar
from paragen.datasets.in_memory_dataset import InMemoryDataset

from .streaming_multi_parallel_text_dataset import MultiSourceOneSample


@register_dataset
class MultiParallelTextDataset(InMemoryDataset):
    """
    Tackle with io and create multiple parallel data

    Args:
        path: a dict of data with their path. `path` can be `None` to build the process pipeline only.
        sort_samples (bool): sort samples before running a task.
            It would be useful in inference without degrading performance.
        max_size: maximum size of loaded data
    """

    def __init__(self,
                 path: Dict[str, str] = None,
                 sort_samples: bool = False,
                 max_size: int = 0,
                 mono = False):
        super().__init__(path, sort_samples=sort_samples, max_size=max_size)
        self._sources = []
        self._mono = mono
        for key in list(path.keys()):
            if not (self._mono and '_tgt' in key):
                self._sources.append(key)

    def _callback(self, sample):
        """
        Callback for parallel data

        Args:
            sample: data in raw format

        Returns:
            sample (dict): a dict of samples consisting of parallel data of different sources
        """
        if self._preprocessed:
            #tmp = {key: eval(val.rsplit('\t', 1)[1]) for key, val in sample.sample.items()}
            #tmp.update({f"{key}_text": val.rsplit('\t', 1)[0] for key, val in sample.sample.items()})
            try:
                sample.sample = {key: eval(val) for key, val in sample.sample.items()}
            except:
                sample.sample = {key: [int(token_id) for token_id in val.split()] for key, val in sample.sample.items()}
        return sample
        
    def _load(self):
        """
        Preload all the data into memory. In the loading process, data are collate_fnd and sorted.
        """
        ori_fin = [GFile(self._path[src]) for src in self._sources]
        fin = zip(*ori_fin)
        self._data = []
        accepted, discarded = 0, 0
        for i, sample in enumerate(progress_bar(fin, streaming=True, desc='Loading Samples')):
            if 0 < self._max_size <= i:
                break
            try:
                if self._mono:
                    sample = self._full_callback(MultiSourceOneSample(key=self._sources[0]+'-'+self._sources[0]+'_tgt', sample={self._sources[0]: sample[0], self._sources[0]+"_tgt": sample[0]}))
                else:
                    sample = self._full_callback(MultiSourceOneSample(key='-'.join(self._sources), sample={src: s.strip('\n') for src, s in zip(self._sources, sample)}))
                self._data.append(sample)
                accepted += 1
            except Exception as e:
                env = Environment()
                if env.debug:
                    raise e
                logger.warning('sample {} is discarded'.format(sample))
                discarded += 1
        # if self._sort_samples:
        #     self._data.sort(key=lambda x: count_sample_token(x))
        self._length = len(self._data)
        logger.info(f'Totally accept {accepted} samples, discard {discarded} samples')
        for fin in ori_fin:
            fin.close()
        self._collate_fn = None
