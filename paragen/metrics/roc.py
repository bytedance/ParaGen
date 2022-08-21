import numpy as np

from paragen.metrics import PairwiseMetric, register_metric
from sklearn.metrics import roc_auc_score

@register_metric
class ROC_AUC(PairwiseMetric):
    """
    Accuracy evaluates accuracy of produced hypotheses labels by comparing with references.
    """

    def __init__(self, is_labeling=False):
        super().__init__()
        self._is_labeling = is_labeling

    def eval(self):
        """
        Calculate the accuracy of produced hypotheses comparing with references
        Returns:
            score (float): evaluation score
        """
        if self._score is not None:
            return self._score
        else:
            if self._is_labeling:
                hypotoken, reftoken = [], []
                for hypo, ref in zip(self.hypos, self.refs):
                    hypotoken.extend(hypo)
                    reftoken.extend(ref)
            else:
                reftoken, hypotoken = self.refs, self.hypos
            self._score = roc_auc_score(reftoken, hypotoken)
        return self._score
