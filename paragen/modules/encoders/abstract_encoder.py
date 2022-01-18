from torch.nn import Module


class AbstractEncoder(Module):
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
        Build encoder with task instance
        """
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """
        Process forward of encoder. Outputs are cached until the encoder is reset.
        """
        if self._mode == 'train':
            if 'out' not in self._cache:
                out = self._forward(*args, **kwargs)
                self._cache['out'] = out
            return self._cache['out']
        else:
            return self._forward(*args, **kwargs)

    def _forward(self, *args, **kwargs):
        """
        Forward function to override. Its results can be auto cached in forward.
        """
        raise NotImplementedError

    @property
    def name(self):
        return self._name

    @property
    def d_model(self):
        raise NotImplementedError

    @property
    def out_dim(self):
        raise NotImplementedError

    def _cache_states(self, name, state):
        """
        Cache a state into encoder cache

        Args:
            name: state key
            state: state value
        """
        self._cache[name] = state

    def reset(self, mode):
        """
        Reset encoder and switch running mode

        Args:
            mode: running mode in [train, valid, infer]
        """
        self._cache.clear()
        self._mode = mode

    def set_cache(self, cache):
        """
        Set cache from outside

        Args:
            cache: cache dict from outside
        """
        self._cache = cache

    def get_cache(self):
        """
        Retrieve inner cache

        Returns:
            - cached states as a Dict
        """
        return self._cache
