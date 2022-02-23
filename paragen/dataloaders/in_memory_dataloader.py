from typing import List
import logging
logger = logging.getLogger(__name__)

from paragen.dataloaders import AbstractDataLoader, register_dataloader
from paragen.datasets.abstract_dataset import AbstractDataset
from paragen.samplers.abstract_sampler import AbstractSampler
from paragen.samplers.distributed_sampler import DistributedSampler


@register_dataloader
class InMemoryDataLoader(AbstractDataLoader):
    """
    InMemoryDataLoader targets to sample and process data from InMemoryDataset

    Args:
        dataset (paragen.datasets.AbstractDataset): source dataset to load.
        sampler (paragen.samplers.AbstractSampler): sampler to retrieve data from the dataset with customized strategy.
        collate_fn: data process pipeline embedded in torch.utils.data.DataLoader
        post_collate_fn: data process pipeline after torch.utils.data.DataLoader,
            which can be adjusted withing a training epoch.
    """

    def __init__(self,
                 dataset: AbstractDataset,
                 sampler: AbstractSampler,
                 cached_samples: List = None,
                 collate_fn=None,
                 post_collate_fn=None,
                 use_cache=False,
                 **kwargs):
        super().__init__(dataset,
                         sampler=None,
                         batch_sampler=sampler.batch_sampler,
                         collate_fn=collate_fn,
                         post_collate_fn=post_collate_fn,
                         **kwargs)
        self._use_cache = use_cache
        if self._use_cache and cached_samples is not None:
            logger.info(f'cached samples (size={len(cached_samples)}) detected, skip init')
            self._cached_samples = cached_samples
            self._has_cached_samples = True
        else:
            self._cached_samples = []
            self._has_cached_samples = False
        self._sampler = sampler
        self._kwargs = kwargs

    def reset(self, epoch=0, *args, **kwargs):
        """
        Reset dataloader
        In torch, parameters of dataloader cannot be modified. Here we reset by re-build a new DataLoader with the same
        parameters.

        Args:
            epoch: training epoch
            step: training step

        Returns:
            dataloader (paragen.dataloaders.AbstractDataLoader): re-build a new DataLoader with possibly new collate_fn
        """
        if len(self._cached_samples) > 0:
            self.dataset.finalize()
            self._sampler.finalize()
            self._has_cached_samples = True
            return self
        else:
            self.dataset.reset()
            self._sampler.reset(epoch)
            return InMemoryDataLoader(self.dataset,
                                      sampler=self._sampler,
                                      cached_samples=None,
                                      collate_fn=self.collate_fn,
                                      post_collate_fn=self._post_collate_fn,
                                      use_cache=self._use_cache,
                                      **self._kwargs)

    def step_update(self, step, states=None):
        """
        Step-level updating on training states

        Args:
            step: learning steps
            states: states recorded in training process, and is used to update `sampler'
        """
        self._sampler.step_update(step, states)

    def epoch_update(self, epoch, states=None):
        """
        Epoch-level updating on training states

        Args:
            epoch: learning epoch
            states: states recorded in training process, and is used to update `sampler'
        """
        self._sampler.epoch_update(epoch, states)

    def __iter__(self):
        """
        Wrap the original data loading process with `post_collate`

        Returns:
            samples: a list of sample with `post_collate` process
        """
        if self._has_cached_samples:
            from paragen.samplers.bucket_sampler import BucketSampler
            if self._sampler.__class__ is BucketSampler or \
                (self._sampler.__class__ is DistributedSampler and
                    self._sampler.sampler.__class__ is BucketSampler):
                import random
                random.shuffle(self._cached_samples)
            for samples in self._cached_samples:
                yield samples
        else:
            for samples in super().__iter__():
                samples = self._callback(samples)
                if self._use_cache:
                    self._cached_samples.append(samples)
                yield samples

    def finalize(self):
        """
        finalization
        """
        self._dataset.finalize()
        self._sampler.finalize()

    def __len__(self):
        return len(self._cached_samples) if self._has_cached_samples else super().__len__()
