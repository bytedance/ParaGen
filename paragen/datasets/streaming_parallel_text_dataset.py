from typing import Dict

from paragen.datasets import register_dataset
from paragen.datasets.streaming_dataset import StreamingDataset
from paragen.utils.io import UniIO


@register_dataset
class StreamingParallelTextDataset(StreamingDataset):
    """
    StreamingParallelTextDataset is a streaming dataset for reading data saved in parallel files.

    Args:
        path: a dict of data with their path. `path` can be `None` to build the process pipeline only.
    """

    def __init__(self,
                 path: Dict[str, str] = None,):
        super().__init__(path)
        self._sources = path.keys()
        self._ori_fin = None

    def build(self, collate_fn=None, preprocessed=False):
        """
        Build input stream

        Args:
             collate_fn: callback defined by a specific task
             preprocessed: whether the data has been preprocessed
        """
        self._collate_fn = collate_fn
        self._preprocessed = preprocessed

        if self._path:
            self._ori_fin = [UniIO(self._path[src]) for src in self._sources]
            self._fin = zip(*self._ori_fin)

    def __iter__(self):
        """
        fetch next sample

        Returns:
            sample: next sample
        """
        for sample in self._fin:
            sample = self._full_callback({src: s.strip('\n') for src, s in zip(self._sources, sample)})
            yield sample

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

    def finalize(self):
        """
        Finalize dataset after finish reading
        """
        for fin in self._ori_fin:
            fin.close()

    def reset(self):
        """
        reset the dataset
        """
        self._pos = 0
        self._ori_fin = [UniIO(self._path[src]) for src in self._sources]
        self._fin = zip(*self._ori_fin)

