from torch.nn import Module


class AbstractSearch(Module):
    """
    AbstractSearch is search algorithm on original neural model to perform special inference.
    """

    def __init__(self):
        super().__init__()
        self._mode = 'infer'

    def build(self, *args, **kwargs):
        """
        Build search algorithm with task instance
        """
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """
        Process forward of search algorithm.
        """
        raise NotImplementedError

    def reset(self, mode):
        """
        Reset encoder and switch running mode

        Args:
            mode: running mode in [train, valid, infer]
        """
        self._mode = mode
