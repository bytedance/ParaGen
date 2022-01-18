import torch.nn as nn


class AbstractEncoderLayer(nn.Module):
    """
    AbstractEncoderLayer is an abstract class for encoder layers.
    """

    def __init__(self):
        super().__init__()
        self._cache = {}
        self._mode = 'train'

    def reset(self, mode):
        """
        Reset encoder layer and switch running mode

        Args:
            mode: running mode in [train, valid, infer]
        """
        self._mode = mode
        self._cache.clear()

    def _update_cache(self, *args, **kwargs):
        """
        Update internal cache from outside states
        """
        pass

    def get_cache(self):
        """
        Retrieve inner cache

        Returns:
            - cached states as a Dict
        """
        return self._cache

    def set_cache(self, cache):
        """
        Set cache from outside

        Args:
            cache: cache dict from outside
        """
        self._cache = cache
