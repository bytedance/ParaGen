from paragen.metrics import PairwiseMetric, register_metric


@register_metric
class Accuracy(PairwiseMetric):
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
            for hypo, ref in zip(self.hypos, self.refs):
                correct += 1 if hypo == ref else 0
            self._score = correct / len(self.hypos)
        return self._score
