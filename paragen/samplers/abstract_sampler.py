import logging
logger = logging.getLogger(__name__)

from torch.utils.data import Sampler

from paragen.utils.data import count_sample_token
from paragen.utils.runtime import progress_bar


class AbstractSampler(Sampler):
    """
    Sampler produces the sample indexes for sequential training.
    Any strategy can be used to produces a permutation list to define the order to fetch samples.

    Args:
        max_samples: maximum samples within a batch
        max_tokens: maximum tokens withint a batch
    """

    def __init__(self, max_samples=None, max_tokens=None):
        super().__init__(None)
        self._max_samples = max_samples
        self._max_tokens = max_tokens

        assert max_samples is not None or max_tokens is not None, \
            'must provide at least one of max_samples and max_tokens'

        self._data_source = None
        self._length = -1
        self._permutation = []

    def __iter__(self):
        """
        Get iterator of the sampler
        """
        return iter(self._permutation)

    def __len__(self):
        """
        Sampler size or dataset capacity
        """
        return self._length

    def __getitem__(self, idx):
        """
        Get the idx index

        Args:
            idx: index in permutation
        """
        return self._permutation[idx]

    def build(self, *args, **kwargs):
        """
        Build Sampler
        """
        raise NotImplementedError

    def reset(self, *args, **kwargs):
        """
        Reset sampler states
        """
        pass

    def step_update(self, *args, **kwargs):
        """
        Update sampler states by steps
        """
        pass

    def epoch_update(self, *args, **kwargs):
        """
        Update sampler states by epochs
        """
        pass

    def finalize(self):
        """
        Finalize sampler after finishing sampling process
        """
        try:
            del self._permutation
            del self._data_source
        except:
            pass

    @property
    def batch_sampler(self):
        """
        Pre-calculate batches within sampler with strategy

        Returns:
            batches: a list of batches of index
        """
        batches = []
        sample_cnt, max_cnt = 0, 0
        batch = []
        self._data_source.reset()
        for i in progress_bar(self._permutation, desc='Calculating Batch Samples'):
            sample = self._data_source[i]
            tc = sample['token_num'] if 'token_num' in sample else count_sample_token(sample)
            if (self._max_tokens and max(max_cnt, tc) * (sample_cnt + 1) > self._max_tokens) \
                    or (self._max_samples and sample_cnt + 1 > self._max_samples):
                batches.append(batch)
                batch, max_cnt, sample_cnt = [], 0, 0
            batch.append(i)
            max_cnt = max(max_cnt, tc)
            sample_cnt += 1
        batches.append(batch)
        logger.info('Deriving total {} batches'.format(len(batches)))
        return batches

    @property
    def max_tokens(self):
        return self._max_tokens

    @property
    def max_samples(self):
        return self._max_samples
