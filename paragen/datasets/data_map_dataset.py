import json
import logging
logger = logging.getLogger(__name__)

from paragen.datasets import register_dataset, AbstractDataset
from paragen.utils.data import possible_load_json
from paragen.utils.io import UniIO


@register_dataset
class DataMapDataset(AbstractDataset):
    """
    A data-map dataset which load data map into memory before running task.
    Then data are fetched from disk according to the idx map.
    DataMapDataset is suitable for dataset of relatively high capacity.

    Args:
        path: data path to read
        data_map_path: path to load data map
        sort_samples (bool): sort samples before running a task.
            It would be useful in inference without degrading performance.
        max_size: maximum size of loaded data
    """

    def __init__(self,
                 path,
                 data_map_path,
                 sort_samples=True,
                 max_size=0):
        super().__init__(path, max_size=max_size)
        self._data_map_path = data_map_path
        self._data_map = None
        self._sort_samples = sort_samples
        self._fin = None

    def build(self, collate_fn=None, preprocessed=False, **kwargs):
        """
        Build input stream and load data into memory

        Args:
             collate_fn: data_collate_function defined by a specific task
             preprocessed: the dataset has been preprocessed
        """
        self._collate_fn = collate_fn
        self._preprocessed = preprocessed

        if self._path:
            self._load()
        self._pos = 0
        self._fin = UniIO(self._path)

    def _load(self):
        """
        Preload the data map into memory.
        """
        with UniIO(self._data_map_path) as fin:
            # MemoryMap is a List[List], namely for each sample [id, position, token_num]
            self._data_map = json.load(fin)
        self._length = len(self._data_map)
        if self._sort_samples:
            self._data_map.sort(key=lambda x: x[-1])

    def __next__(self):
        """
        fetch next sample

        Returns:
            sample: next sample
        """
        success, sample = False, None
        while not success:
            if self._length is not None and self._pos >= self._length:
                raise StopIteration
            self._pos += 1
            self._fin.seek(self._data_map[self._pos][1])
            sample = next(self._fin)
            try:
                sample = self._full_callback(sample)
                success = True
            except Exception as e:
                logger.warning(e)
        return sample

    def __getitem__(self, idx):
        """
        get idx-th item within data
        Args:
            idx: index of data

        Returns:
            sample at data_map[idx]
        """
        self._fin.seek(self._data_map[self._pos][1])
        sample = next(self._fin)
        sample = self._full_callback(sample)
        return sample

    def _callback(self, sample):
        """
        json.load data

        Args:
            sample: data in raw format

        Returns:
            sample (dict): a dict of samples consisting of parallel data of different sources
        """
        return possible_load_json(sample)
