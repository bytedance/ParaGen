from rouge import Rouge as Rg

from paragen.metrics import PairwiseMetric, register_metric


@register_metric
class Rouge(PairwiseMetric):
    """
    Rouge evaluates rouge scores of produced hypotheses by comparing with references.

    Args:
        ngram: ['1', '2', 'l'] stands for ['rouge-1', 'rouge-2', 'rouge-l']
    """

    def __init__(self, ngram='1'):
        super().__init__()
        self._ngram = ngram
        self._rouge = Rg()

    def eval(self):
        """
        Evaluate the performance with buffered hypotheses and references.
        """
        if self._score is not None:
            return self._score
        else:
            score = self._rouge.get_scores(self._hypos, self._refs, avg=True)
            self._score = score[f'rouge-{self._ngram}']['f']
        return self._score
