from paragen.metrics import AbstractMetric


class PairwiseMetric(AbstractMetric):
    """
    PairwiseMtric evaluates pairwise comparison between refs and hypos.
    """

    def __init__(self):
        super().__init__()
        self._hypos, self._refs = [], []

    def reset(self):
        """
        Reset metric for a new round of evaluation
        """
        self._hypos.clear()
        self._refs.clear()
        self._score = None

    def add_all(self, hypos, refs):
        """
        Add all hypos and refs
        """
        for hypo, ref in zip(hypos, refs):
            self.add(hypo, ref)

    def add(self, hypo, ref):
        """
        Add parallel hypotheses and references to metric buffer
        """
        self._hypos.append(hypo)
        self._refs.append(ref)

    def eval(self):
        """
        Evaluate the performance with buffered hypotheses and references.
        """
        raise NotImplementedError

    def __len__(self):
        return len(self._hypos)

    def __getitem__(self, idx):
        return self._hypos[idx], self._refs[idx]

    def get_item(self, idx, to_str=False):
        """
        fetch a item at given index

        Args:
            idx: index of a pair of hypothesis and reference
            to_str: transform the pair to str format before return it

        Returns:
            item: a pair of item in tuple or string format
        """
        ret = self[idx]
        if to_str:
            ret = '\n\tHypothesis: {}\n\tGround Truth: {}\n'.format(ret[0], ret[1])
        return ret

    @property
    def hypos(self):
        return self._hypos

    @property
    def refs(self):
        return self._refs
