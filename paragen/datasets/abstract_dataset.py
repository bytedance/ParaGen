import json

from torch.utils.data import Dataset as D

from paragen.utils.io import create_data_map, UniIO
from paragen.utils.runtime import progress_bar


class AbstractDataset(D):
    """
    Tackle with io and create parallel data

    Args:
        path: data path to read
        max_size: maximum size of loaded data
    """

    def __init__(self,
                 path=None,
                 max_size=0,):
        self._path = path
        self._max_size = max_size

        self._collate_fn = None
        self._preprocessed = False
        self._pos, self._length = 0, None

    def build(self, *args, **kwargs):
        """
        Build input stream for all data set in path
        """
        raise NotImplementedError

    def reset(self, *args, **kwargs):
        """
        Reset dataset
        """
        pass

    def shuffle(self, *args, **kwargs):
        """
        shuffle data
        """
        pass

    def finalize(self):
        """
        Finalize dataset after finish reading
        """
        pass

    def __getitem__(self, index):
        """
        fetch an item with index

        Args:
            index: index of item to fetch
        """
        raise NotImplementedError

    def __len__(self):
        """
        length of datasset

        Returns:
            length: data size
        """
        return self._length

    def __next__(self):
        """
        fetch next tokenized sample

        Returns:
            sample: next sample
        """
        raise NotImplementedError

    def __iter__(self):
        """
        get an iterator of dataset

        Returns:
            iterator: dataset as an iterator
        """
        return self

    def _callback(self, sample):
        """
        Intrinsic callback determined by data format

        Args:
            sample: a sample of raw data

        Returns:
            sample: a parsed sample in straight forward format, e.g. str or dict
        """
        return sample

    def _full_callback(self, sample):
        """
        combine intrinsic callback and data collation

        Args:
            sample: a read-in sample

        Returns:
            sample: preprocessed sample
        """
        sample = self._callback(sample)
        if self._collate_fn is not None and not self._preprocessed:
            sample = self._collate_fn(sample)
        return sample

    def write(self, path, data_map_path=None):
        """
        Read, index and dump raw data to a target path

        Args:
            path: path to save preprocessed data
            data_map_path: path to store memory map
        """
        with UniIO(path, 'w') as fout:
            for sample in progress_bar(self):
                fout.write('{}\n'.format(json.dumps(sample)))
        if data_map_path:
            data_map = create_data_map(path)
            with open(data_map_path, 'w') as fout:
                json.dump(data_map, fout)
