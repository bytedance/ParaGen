import random

from paragen.samplers import AbstractSampler, register_sampler
from paragen.utils.runtime import Environment


@register_sampler
class ShuffleSampler(AbstractSampler):
    """
    ShuffleSampler shuffle the order before fetching samples.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._env = Environment()

    def build(self, data_source):
        """
        Build sampler over data_source

        Args:
            data_source: a list of data
        """
        self._data_source = data_source
        self._permutation = [_ for _ in range(len(self._data_source))]
        self._length = len(self._permutation)
        self.reset(0)

    def reset(self, epoch, *args, **kwargs):
        """
        Resetting sampler states / shuffle reading order for next round of iteration
        """
        random.seed(self._env.seed + epoch)
        random.shuffle(self._permutation)
