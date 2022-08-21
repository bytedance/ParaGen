import numpy as np

from paragen.metrics import PairwiseMetric, register_metric


@register_metric
class TokenAccuracy(PairwiseMetric):
    """
    Accuracy evaluates accuracy of produced hypotheses labels by comparing with references.
    """

    def __init__(self):
        super().__init__()

    def eval(self):
        """
        Calculate the accuracy of produced hypotheses comparing with references
        Returns:
            score (float): evaluation score
        """
        if self._score is not None:
            return self._score
        else:
            correct = 0
            tot = 0
            hypos, refs = self.hypos, self.refs
            if isinstance(self.hypos[0], str):
                hypos = [hypo.split() for hypo in self.hypos]
            if isinstance(self.refs[0], str):
                refs = [ref.split() for ref in self.refs]
            for hypo, ref in zip(hypos, refs):
                if len(ref) == 0:
                    continue
                minlen = min(len(ref), len(hypo))
                if not isinstance(ref[0], str):
                    ref = [1 if r >= 0.5 else 0 for r in ref[:minlen]]
                    hypo = [1 if h >= 0.5 else 0 for h in hypo[:minlen]]
                else:
                    ref = ref[:minlen]
                    hypo = hypo[:minlen]
                correct += (np.array(hypo) == np.array(ref)).sum()
                tot += len(ref)
            self._score = correct / tot
        return self._score
