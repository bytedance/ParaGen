import random

from paragen.samplers import AbstractSampler, register_sampler
from paragen.utils.runtime import Environment


@register_sampler
class BucketSampler(AbstractSampler):
    """
    BucketSampler put samples of similar size into a bucket to lift computational efficiency and to accelerate training.

    Args:
        noise: inject noise when create buckets for each iteration.
    """

    def __init__(self, noise=0., **kwargs):
        super().__init__(**kwargs)
        self._noise = noise

    def build(self, data_source):
        """
        Build sampler over data_source

        Args:
            data_source: a list of data
        """
        self._data_source = data_source
        self._length = len(self._data_source)
        self.reset(0)

    def reset(self, epoch, *args, **kwargs):
        """
        Resetting sampler states / shuffle reading order for next round of iteration

        Args:
            epoch: iteration epoch
        """
        env = Environment()
        random.seed(env.seed + epoch)
        token_nums = [(i, self._inject_noise(sample['token_num'])) for i, sample in enumerate(self._data_source)]
        token_nums.sort(key=lambda x: x[1], reverse=True)
        self._permutation = [idx for idx, _ in token_nums]

    @property
    def batch_sampler(self):
        """
        Pre-calculate batches within sampler with strategy

        Returns:
            batches: a list of batches of index
        """
        batches = super().batch_sampler
        random.shuffle(batches)
        return batches

    def _inject_noise(self, x):
        """
        Disturb size

        Args:
            x: size

        Returns:
            disturbed size
        """
        if self._noise > 0:
            variance = int(x * self._noise)
            r = random.randint(-variance, variance)
            return x + r
        else:
            return x
