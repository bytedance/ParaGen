import math

from torch.utils.data import Sampler

from paragen.samplers import AbstractSampler
from paragen.utils.runtime import Environment


class DistributedSampler(Sampler):
    """
    DistributedSampler wraps a sampler for distributed training.

    Args:
        sampler (paragen.samplers.AbstractSampler): a Sampler instance
    """

    def __init__(self, sampler: AbstractSampler):
        super().__init__(None)
        self._sampler = sampler

        self._max_tokens, self._max_samples = sampler.max_tokens, sampler.max_samples
        self._data_source = None
        self._num_replicas = None
        self._rank = None
        self._num_samples = None
        self._indices = None

    def build(self, data_source):
        """
        Build sampler over data_source

        Args:
            data_source: a list of data
        """
        self._data_source = data_source

        env = Environment()
        self._num_replicas = env.distributed_world
        self._rank = env.rank
        self._num_samples = int(math.ceil(len(self._data_source) * 1.0 / self._num_replicas))
        self._sampler.build(data_source)

        indices = list(range(len(self._sampler)))
        total_size = self._num_samples * self._num_replicas
        # add extra samples to make it evenly divisible
        indices += indices[:(total_size - len(indices))]
        # subsample
        indices = indices[self._rank:total_size:self._num_replicas]
        indices = [self._sampler[i] for i in indices]
        self._indices = indices

    def __iter__(self):
        return iter(self._indices)

    def __len__(self):
        return self._num_samples

    def __getitem__(self, idx):
        return self._indices[idx]

    @property
    def batch_sampler(self):
        """
        Pre-calculate batches within sampler with strategy

        Returns:
            batches: a list of batches of index
        """
        batches = self._sampler.batch_sampler
        num_batches = int(math.ceil(len(batches) * 1.0 / self._num_replicas))
        tot_num_batches = num_batches * self._num_replicas
        batches += batches[:(tot_num_batches - len(batches))]
        batches = batches[self._rank:tot_num_batches:self._num_replicas]
        return batches

    def reset(self, *args, **kwargs):
        """
        Reset sampler states
        """
        self._sampler.reset(*args, **kwargs)

    def step_update(self, *args, **kwargs):
        """
        Update sampler states by steps
        """
        self._sampler.step_update(*args, **kwargs)

    def epoch_update(self, *args, **kwargs):
        """
        Update sampler states by epochs
        """
        self._sampler.epoch_update(*args, **kwargs)

    def finalize(self):
        """
        Finalize sampler after finishing sampling process
        """
        self._sampler.finalize()

    @property
    def max_tokens(self):
        return self._max_tokens

    @property
    def max_samples(self):
        return self._max_samples

    @property
    def sampler(self):
        return self._sampler
