import random

from paragen.datasets import AbstractDataset


class InMemoryDataset(AbstractDataset):
    """
    An in-memory dataset which load data in memory before running task.
    InMemoryDataset is suitable for dataset of relatively low capacity.

    Args:
        path: data path to read
        sort_samples (bool): sort samples before running a task.
            It would be useful in inference without degrading performance.
        max_size: maximum size of loaded data
    """

    def __init__(self,
                 path,
                 sort_samples=False,
                 max_size=0,):
        super().__init__(path, max_size=max_size)

        self._data = None
        self._sort_samples = sort_samples

    def build(self, collate_fn=None, preprocessed=False, **kwargs):
        """
        Build input stream and load data into memory

        Args:
             collate_fn: callback defined by a specific task
             preprocessed: data has been preprocessed
        """
        self._collate_fn = collate_fn
        self._preprocessed = preprocessed

        if self._path:
            self._load()
        self._pos = 0

    def _load(self):
        """
        Load data into memory
        """
        raise NotImplementedError

    def shuffle(self):
        """
        shuffle preload data
        """
        random.shuffle(self._data)

    def __getitem__(self, index):
        """
        fetch an item at index

        Args:
            index: index of item to fetch

        Returns:
            sample: data of index in preload data list
        """
        return self._data[index]

    def __iter__(self):
        for sample in self._data:
            yield sample

    def __next__(self):
        """
        fetch next sample

        Returns:
            sample: next sample
        """
        if self._pos < len(self._data):
            sample = self._data[self._pos]
            self._pos += 1
        else:
            raise StopIteration
        return sample

    def reset(self):
        """
        Reset io for a new round of iteration
        """
        self._pos = 0

    def finalize(self):
        super().finalize()
        try:
            del self._data
        except:
            pass

