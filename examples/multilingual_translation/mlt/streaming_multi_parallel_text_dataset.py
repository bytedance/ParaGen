from typing import Dict
from tensorflow.io.gfile import GFile
from dataclasses import dataclass

from paragen.datasets import register_dataset
from paragen.datasets.streaming_dataset import StreamingDataset


@dataclass
class MultiSourceOneSample(object):
    key: str
    sample: str


@register_dataset
class StreamingMultiParallelTextDataset(StreamingDataset):
    """
    Tackle with io and create parallel data

    Args:
        path: a dict of data with their path. `path` can be `None` to build the process pipeline only.

    """

    def __init__(self,
                 path: Dict[str, str] = None,
                 mono = False):
        super().__init__(path)
        self._sources = path
        self._ori_fin = None
        self._mono = mono
        if self._mono:
            for key in self._sources.keys():
                if '_tgt' in key:
                    del self._sources[key]
                    break

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

    def _callback(self, sample):
        """
        Callback for parallel data

        Args:
            sample: data in raw format

        Returns:
            sample (dict): a dict of samples consisting of parallel data of different sources
        """
        if self._preprocessed:
            try:
                sample.sample = {key: eval(val) for key, val in sample.sample.items()}
            except:
                sample.sample = {key: [int(token_id) for token_id in val.split()] for key, val in sample.sample.items()}
        return sample

    def build(self, collate_fn=None, preprocessed=False):
        """
        Build input stream

        Args:
             collate_fn: callback defined by a specific task
             preprocessed: whether the data has been preprocessed
        """
        self._sources = list(self._sources.keys())
        self._collate_fn = collate_fn
        self._preprocessed = preprocessed

        if self._path:
            self._ori_fin = [GFile(self._path[src], 'r') for src in self._sources]
            self._fin = zip(*self._ori_fin)

    def __next__(self):
        """
        fetch next sample

        Returns:
            sample: next sample
        """
        self._pos += 1
        if self._length is not None and self._pos >= self._length:
            raise StopIteration
        sample = next(self._fin)
        if self._mono:
            sample = self._full_callback(MultiSourceOneSample(key=self._sources[0]+'-'+self._sources[0]+'_tgt', sample={self._sources[0]: sample[0], self._sources[0]+"_tgt": sample[0]}))
        else:
            sample = self._full_callback(MultiSourceOneSample(key='-'.join(self._sources), sample={src: s.strip('\n') for src, s in zip(self._sources, sample)}))

        return sample

    def reset(self):
        """
        reset the dataset
        """
        self._pos = 0
        for fin in self._ori_fin:
            fin.seek(0)
            
    def finalize(self):
        """
        Finalize dataset after finish reading
        """
        for fin in self._ori_fin:
            fin.close()

