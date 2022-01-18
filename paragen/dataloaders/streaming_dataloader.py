import random

from paragen.dataloaders import AbstractDataLoader, register_dataloader
from paragen.datasets.abstract_dataset import AbstractDataset
from paragen.utils.ops import search_key
from paragen.utils.runtime import Environment


@register_dataloader
class StreamingDataLoader(AbstractDataLoader):
    """
    StreamingDataLoader target to sample and process data from StreamingDataset

    Args:
        dataset (paragen.datasets.AbstractDataset): source dataset to load.
        collate_fn: data process pipeline embedded in torch.utils.data.DataLoader
        post_collate_fn: data process pipeline after torch.utils.data.DataLoader,
            which can be adjusted withing a training epoch.
        max_samples: number of samples within a batch, namely batch size
        batch_size: number of samples within a batch
        max_tokens: create batch with max tokens
        maxlen: maximum length for creating buffered buckets
        length_interval: interval between buckets
        max_shuffle_size: maximum size of a bucket for batch shuffling
    """

    def __init__(self,
                 dataset: AbstractDataset,
                 collate_fn=None,
                 post_collate_fn=None,
                 max_samples=None,
                 batch_size=None,
                 max_tokens=None,
                 maxlen=None,
                 length_interval=8,
                 max_shuffle_size=1,
                 **kwargs):
        assert 'sampler' not in kwargs, 'do not specifying sampler when using StreamingDataLoader'
        if 'num_workers' in kwargs:
            # samples are pre-computed for each process. it may cause problem when num_workers > 1
            assert kwargs['num_workers'] in [0, 1]
        if max_samples is not None:
            batch_size = max_samples
        if max_tokens is not None:
            assert collate_fn is None, 'use post_collate_fn only when using max token'
            assert batch_size is None, 'DO NOT specify batch_size when using max token'
        self._max_tokens = max_tokens
        kwargs['batch_size'] = batch_size
        super().__init__(dataset,
                         collate_fn=collate_fn,
                         post_collate_fn=post_collate_fn,
                         **kwargs)
        self._kwargs = kwargs

        env = Environment()
        self._distributed_wolrds = env.distributed_world
        self._rank = env.rank

        self._maxlen = maxlen
        self._length_interval = length_interval
        self._max_shuffle_size = max_shuffle_size
        if self._max_tokens is not None:
            if maxlen is None:
                env = Environment()
                self._maxlen = search_key(env.configs, 'maxlen')
            self._bucket_num = self._maxlen // length_interval + 1
            self._max_buffered_batch_num = self._max_shuffle_size * self._distributed_wolrds
            self._max_buffer_sizes = {
                i: max_tokens // ((i + 1) * length_interval) if max_tokens is not None else max_samples
                for i in range(self._bucket_num)
            }
            self._buckets = {i: [] for i in range(self._bucket_num)}

        self._buffered_batches = []

    def reset(self, *args, **kwargs):
        """
        Reset dataloader
        In torch, parameters of dataloader cannot be modified. Here we reset by re-build a new DataLoader with the same
        parameters.

        Returns:
            dataloader (paragen.dataloaders.DataLoader): re-build a new DataLoader with possibly new collate_fn
        """
        self.dataset.reset()
        return StreamingDataLoader(self.dataset,
                                   collate_fn=self._collate_fn,
                                   post_collate_fn=self._post_collate_fn,
                                   max_tokens=self._max_tokens,
                                   maxlen=self._maxlen,
                                   length_interval=self._length_interval,
                                   max_shuffle_size=self._max_shuffle_size,
                                   **self._kwargs)

    def __iter__(self):
        """
        Wrap the original data loading process with `post_collate`

        Returns:
            samples: a list of sample with `post_collate` process
        """
        iterator = super().__iter__()
        for samples in iterator:
            if self._max_tokens is not None:
                sample = samples
                if sample['token_num'] >= self._maxlen or sample['token_num'] <= 0:
                    continue
                # compute bucket idx
                bucket_idx = (sample['token_num'] - 1) // self._length_interval
                # append sample to its bucket
                bucket = self._buckets[bucket_idx]
                bucket.append(sample)
                if len(bucket) == self._max_buffer_sizes[bucket_idx]:
                    # if bucket size reach its maximum sentences, put it into buffer as a batch
                    self._buffered_batches.append([s for s in bucket])
                    bucket.clear()
                    if len(self._buffered_batches) == self._max_buffered_batch_num:
                        # if bucket size reach maximum shuffle size, do shuffling within buffer
                        random.shuffle(self._buffered_batches)
                        self._buffered_batches = self._buffered_batches[
                                                 self._rank:self._max_buffered_batch_num:self._distributed_wolrds]
                        for s in self._buffered_batches:
                            yield self._callback(s)
                        self._buffered_batches.clear()
            else:
                self._buffered_batches.append(samples)
                if len(self._buffered_batches) == self._distributed_wolrds:
                    yield self._callback(self._buffered_batches[self._rank])
                    self._buffered_batches.clear()
        if self._max_tokens is not None:
            if len(self._buffered_batches) >= self._distributed_wolrds:
                random.shuffle(self._buffered_batches)
                batch_num = len(self._buffered_batches) // self._distributed_wolrds * self._distributed_wolrds
                self._buffered_batches = self._buffered_batches[self._rank:batch_num:self._distributed_wolrds]
                for s in self._buffered_batches:
                    yield self._callback(s)
                self._buffered_batches.clear()

    def __len__(self) -> int:
        return 0
