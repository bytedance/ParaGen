from paragen.samplers import AbstractSampler, register_sampler


@register_sampler
class SequentialSampler(AbstractSampler):
    """
    SequentialSampler iterates on samples sequentially.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, data_source):
        """
        Build sampler over data_source

        Args:
            data_source: a list of data
        """
        self._data_source = data_source
        self._permutation = [_ for _ in range(len(self._data_source))]
        self._length = len(self._permutation)
