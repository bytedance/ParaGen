class AbstractMetric:
    """
    Metric evaluates the performance with produced hypotheses and references.
    """

    def __init__(self):
        self._score = None

    def build(self, *args, **kwargs):
        """
        Build metric
        """
        self.reset()

    def reset(self):
        """
        Reset metric for a new round of evaluation
        """
        pass

    def add_all(self, *args, **kwargs):
        raise NotImplementedError

    def add(self, *args, **kwargs):
        """
        Add parallel hypotheses and references to metric buffer
        """
        raise NotImplementedError

    def eval(self):
        """
        Evaluate the performance with buffered hypotheses and references.
        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def get_item(self, idx, to_str=False):
        """
        fetch a item at given index

        Args:
            idx: index of a pair of hypothesis and reference
            to_str: transform the pair to str format before return it

        Returns:
            item: a pair of item in tuple or string format
        """
        raise NotImplementedError

