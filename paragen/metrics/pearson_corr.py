from scipy.stats import pearsonr
import numpy as np

from paragen.metrics import PairwiseMetric, register_metric


@register_metric
class PearsonCorr(PairwiseMetric):
    """
    PearsonCorr evaluates pearson's correlation of produced hypotheses labels by comparing with references.
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
            self._score = pearsonr(np.array(self._hypos), np.array(self._refs))[0]
        return self._score

