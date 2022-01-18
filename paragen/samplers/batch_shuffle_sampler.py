import random

from paragen.samplers import register_sampler
from paragen.samplers.sequential_sampler import SequentialSampler


@register_sampler
class BatchShuffleSampler(SequentialSampler):
    """
    BatchShuffleSampler pre-compute all the batch sequentially,
    and shuffle the reading order of batches before an new round of iteration.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
