from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import numpy as np

from paragen.metrics import PairwiseMetric, register_metric


@register_metric
class R2(PairwiseMetric):
    """
    SpearmanCorr evaluates spearman's correlation of produced hypotheses labels by comparing with references.
    """

    def __init__(self):
        super().__init__()

    def eval(self):
        """
        Calculate the R2 correlation of produced hypotheses comparing with references
        Returns:
            score (float): evaluation score
        """
        if self._score is not None:
            return self._score
        else:
            x, y = np.array(self._hypos), np.array(self._refs)
            reg = LinearRegression().fit(x.reshape(-1, 1), y)
            y_pred = reg.predict(x.reshape(-1, 1))
            self._score = r2_score(y, y_pred)
        return self._score
