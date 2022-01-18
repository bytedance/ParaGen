from sklearn.metrics import matthews_corrcoef

from paragen.metrics import PairwiseMetric, register_metric


@register_metric
class MatthewsCorr(PairwiseMetric):
    """
    MatthewsCorr evaluates matthews correlation of produced hypotheses labels by comparing with references.
    """

    def __init__(self):
        super().__init__()

    def eval(self):
        """
        Calculate the spearman correlation of produced hypotheses comparing with references
        Returns:
            score (float): evaluation score
        """
        if self._score is not None:
            return self._score
        else:
            self._score = matthews_corrcoef(self._refs, self._hypos)
        return self._score
