import json
import logging
logger = logging.getLogger(__name__)

from paragen.datasets import register_dataset
from paragen.datasets.in_memory_dataset import InMemoryDataset
from paragen.utils.data import count_sample_token
from paragen.utils.runtime import progress_bar


@register_dataset
class TFRecordDataset(InMemoryDataset):
    """
    A tfrecord dataset is an in-memory dataset for reading data in tfrecord

    Args:
        path: data path to read
        index_path: path to load data map
        description: decription for tfrecord
        sort_samples (bool): sort samples before running a task.
            It would be useful in inference without degrading performance.
        max_size: maximum size of loaded data
    """

    def __init__(self,
                 path,
                 index_path=None,
                 description=None,
                 sort_samples=False,
                 max_size=0):
        super().__init__(path, sort_samples=sort_samples, max_size=max_size)
        self._index_path = index_path
        self._description = description

    def _load(self):
        """
        Preload all the data into memory. In the loading process, data are preprocessed and sorted.
        """
        import tfrecord
        fin = tfrecord.tfrecord_loader(self._path,
                                       self._index_path,
                                       self._description)
        self._data = []
        accecpted, discarded = 0, 0
        for i, sample in enumerate(progress_bar(fin, streaming=True, desc='Loading Samples')):
            if 0 < self._max_size <= i:
                break
            try:
                self._data.append(self._full_callback(sample))
                accecpted += 1
            except Exception:
                logger.warning('sample {} is discarded'.format(sample))
                discarded += 1
        if self._sort_samples:
            self._data.sort(key=lambda x: count_sample_token(x))
        self._length = len(self._data)
        logger.info(f'Totally accept {accecpted} samples, discard {discarded} samples')

    def _callback(self, sample):
        """
        Callback for json data

        Args:
            sample: data in raw format

        Returns:
            sample (dict): a dict of samples consisting of parallel data of different sources
        """
        sample = json.loads(sample)
        return sample
