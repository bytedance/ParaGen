from torch.nn import Module


class AbstractDecoder(Module):
    """
    AbstractEncoder is the abstract for encoders, and defines general interface for encoders.

    Args:
        name: encoder name
    """

    def __init__(self, name=None):
        super().__init__()
        self._name = name
        self._cache = {}
        self._mode = 'train'

    def build(self, *args, **kwargs):
        """
        Build decoder with task instance
        """
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """
        Process forward of decoder.
        """
        raise NotImplementedError

    def reset(self, mode):
        """
        Reset encoder and switch running mode

        Args:
            mode: running mode in [train, valid, infer]
        """
        self._cache.clear()
        self._mode = mode

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
