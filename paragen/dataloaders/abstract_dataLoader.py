from torch.utils.data import DataLoader

from paragen.datasets.abstract_dataset import AbstractDataset
from paragen.samplers.abstract_sampler import AbstractSampler
from paragen.utils.ops import echo


class AbstractDataLoader(DataLoader):
    """
    AbstractDataLoader to sample and process data from dataset

    Args:
        dataset (paragen.datasets.AbstractDataset): source dataset to load.
        sampler (paragen.samplers.AbstractSampler): sampler to retrieve data from the dataset with customized strategy.
        collate_fn: data process pipeline embedded in torch.utils.data.DataLoader
        post_collate_fn: data process pipeline after torch.utils.data.DataLoader,
            which can be adjusted withing a training epoch.
        num_workers: number of workers when doing multi-processing
        multiprocessing_context: running context for multiprocessing
    """

    def __init__(self,
                 dataset: AbstractDataset,
                 sampler: AbstractSampler = None,
                 batch_sampler=None,
                 collate_fn=None,
                 post_collate_fn=None,
                 num_workers=0,
                 multiprocessing_context=None,
                 **kwargs):
        self._collate_fn = collate_fn
        super().__init__(dataset,
                         sampler=sampler,
                         batch_sampler=batch_sampler,
                         collate_fn=collate_fn,
                         num_workers=num_workers,
                         multiprocessing_context=multiprocessing_context,
                         **kwargs)
        if collate_fn is None:
            self.collate_fn = echo
        self._post_collate_fn = post_collate_fn

    def reset(self, *args, **kwargs):
        """
        Reset dataloader
        In torch, parameters of dataloader cannot be modified. Here we reset by re-build a new DataLoader with the same
        parameters.

        Returns:
            dataloader (paragen.dataloaders.DataLoader): re-build a new DataLoader with possibly new collate_fn
        """
        pass

    def step_update(self, step, states=None):
        """
        Step-level updating on training states

        Args:
            step: learning steps
            states: states recorded in training process, and is used to update `sampler'
        """
        pass

    def epoch_update(self, epoch, states=None):
        """
        Epoch-level updating on training states

        Args:
            epoch: learning epoch
            states: states recorded in training process, and is used to update `sampler'
        """
        pass

    def _callback(self, samples):
        """
        Callback processing for a batch of samples. It is used for performing post_collate_fn over a batch of samples.

        Args:
            samples: samples fetched from dataloader

        Returns: processed samples
        """
        if self._post_collate_fn is not None:
            if isinstance(samples, dict):
                keys, values = samples.keys(), samples.values()
                values = zip(*values)
                samples = [{k: v for k, v in zip(keys, val)} for val in values]
            samples = self._post_collate_fn(samples)
        return samples

    def finalize(self):
        """
        finalization
        """
        self.sampler.finalize()
