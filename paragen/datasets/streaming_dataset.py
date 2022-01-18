from typing import Dict

from torch.utils.data import IterableDataset

from paragen.datasets import AbstractDataset


class StreamingDataset(AbstractDataset, IterableDataset):
    """
    Tackle with io and create parallel data

    Args:
        path: a dict of data with their path. `path` can be `None` to build the process pipeline only.

    """

    def __init__(self,
                 path: Dict[str, str],):
        super().__init__(path)
        self._fin = None
        self._length = None

    def shuffle(self):
        """
        shuffle preload data
        """
        pass

    def __getitem__(self, index):
        """
        fetch an item with index

        Args:
            index: index of item to fetch

        Returns:
            sample: data of index in preload data list
        """
        return next(self)

    def __next__(self):
        """
        fetch next sample

        Returns:
            sample: next sample
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset io for a new round of iteration
        """
        pass

    def __len__(self):
        """
        Compute dataset length

        Returns:
            dataset length
        """
        return 0

