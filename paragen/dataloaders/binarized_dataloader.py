import json
import math
import random

from paragen.dataloaders import register_dataloader, AbstractDataLoader
from paragen.utils.io import UniIO
from paragen.utils.runtime import Environment
from paragen.utils.tensor import list2tensor


@register_dataloader
class BinarizedDataLoader(AbstractDataLoader):
    """
    AbstractDataLoader to sample and process data from dataset

    Args:
        path: path to load binarized data
    """

    def __init__(self,
                 path,
                 preload=False,
                 length_interval=8,
                 max_shuffle_size=1,
                 **kwargs):
        super().__init__(None)
        self._path = path
        self._preload = preload
        self._batches = None
        self._length_interval = length_interval
        self._max_shuffle_size = max_shuffle_size

        env = Environment()
        self._rank = env.rank
        self._distributed_wolrds = env.distributed_world
        self._max_buffered_batch_num = self._max_shuffle_size * self._distributed_wolrds
        self._buffered_batches = []

        if preload:
            self._batches = []
            with UniIO(self._path) as fin:
                for batch in fin:
                    batch = json.loads(batch)
                    batch = list2tensor(batch)
                    self._batches.append(batch)
            total_size = int(math.ceil(len(self._batches) * 1.0 / self._distributed_wolrds)) * self._distributed_wolrds
            self._batches += self._batches[:(total_size - len(self._batches))]

    def reset(self, *args, **kwargs):
        if not self._preload:
            self._batches = UniIO(self._path)
        else:
            if self._max_shuffle_size > 0:
                random.shuffle(self._batches)
        self._buffered_batches.clear()
        return self

    def __iter__(self):
        for batch in self._batches:
            if not self._preload:
                batch = json.loads(batch)
                batch = list2tensor(batch)
            self._buffered_batches.append(batch)
            if len(self._buffered_batches) == self._max_buffered_batch_num:
                for s in self._dispatch():
                    yield s
        if len(self._buffered_batches) >= self._distributed_wolrds:
            for s in self._dispatch():
                yield s

    def _dispatch(self):
        random.shuffle(self._buffered_batches)
        batch_num = len(self._buffered_batches) // self._distributed_wolrds * self._distributed_wolrds
        self._buffered_batches = self._buffered_batches[self._rank:batch_num:self._distributed_wolrds]
        for s in self._buffered_batches:
            yield s
        self._buffered_batches.clear()

    def __len__(self):
        return len(self._batches) // self._distributed_wolrds if self._preload else 0
