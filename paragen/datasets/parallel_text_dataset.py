from typing import Dict
import logging
logger = logging.getLogger(__name__)

from paragen.datasets import register_dataset
from paragen.datasets.in_memory_dataset import InMemoryDataset
from paragen.utils.data import count_sample_token
from paragen.utils.io import UniIO
from paragen.utils.runtime import progress_bar, Environment


@register_dataset
class ParallelTextDataset(InMemoryDataset):
    """
    ParallelTextDataset is an in-memory dataset for reading data saved in parallel files.

    Args:
        path: a dict of data with their path. `path` can be `None` to build the process pipeline only.
        sort_samples (bool): sort samples before running a task.
            It would be useful in inference without degrading performance.
        max_size: maximum size of loaded data
    """

    def __init__(self,
                 path: Dict[str, str] = None,
                 sort_samples: bool = False,
                 max_size: int = 0,):
        super().__init__(path, sort_samples=sort_samples, max_size=max_size)
        self._sources = list(path.keys())

    def _callback(self, sample):
        """
        Callback for parallel data

        Args:
            sample: data in raw format

        Returns:
            sample (dict): a dict of samples consisting of parallel data of different sources
        """
        if self._preprocessed:
            sample = {key: [eval(v) for v in val] for key, val in sample.items()}
        return sample

    def _load(self):
        """
        Preload all the data into memory. In the loading process, data are collate_fnd and sorted.
        """
        ori_fin = [UniIO(self._path[src]) for src in self._sources]
        fin = zip(*ori_fin)
        self._data = []
        accepted, discarded = 0, 0
        for i, sample in enumerate(progress_bar(fin, streaming=True, desc='Loading Samples')):
            if 0 < self._max_size <= i:
                break
            try:
                sample = self._full_callback({
                    src: s.strip('\n')
                    for src, s in zip(self._sources, sample)
                })
                self._data.append(sample)
                accepted += 1
            except Exception as e:
                env = Environment()
                if env.debug:
                    raise e
                logger.warning('sample {} is discarded'.format(sample))
                discarded += 1
        if self._sort_samples:
            self._data.sort(key=lambda x: count_sample_token(x))
        self._length = len(self._data)
        logger.info(f'Totally accept {accepted} samples, discard {discarded} samples')
        for fin in ori_fin:
            fin.close()

        self._collate_fn = None
