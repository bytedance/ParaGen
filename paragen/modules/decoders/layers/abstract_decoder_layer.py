from typing import Dict

import torch
import torch.nn as nn


class AbstractDecoderLayer(nn.Module):
    """
    AbstractDecoderLayer is an abstract class for decoder layers.
    """

    def __init__(self):
        super().__init__()
        self._cache = dict()
        self._mode = 'train'
        self._dummy_param = nn.Parameter(torch.empty(0))

    def reset(self, mode: str):
        """
        Reset encoder layer and switch running mode

        Args:
            mode: running mode in [train, valid, infer]
        """
        self._cache: Dict[str, torch.Tensor] = {"prev": self._dummy_param}
        self._mode = mode

    def _update_cache(self, *args, **kwargs):
        """
        Update cache with current states
        """
        pass

    def get_cache(self):
        """
        Retrieve inner cache

        Returns:
            - cached states as a Dict
        """
        return self._cache

    def set_cache(self, cache: Dict[str, torch.Tensor]):
        """
        Set cache from outside

        Args:
            cache: cache dict from outside
        """
        self._cache = cache
