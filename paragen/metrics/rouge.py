from rouge import Rouge as RG

from paragen.metrics import PairwiseMetric, register_metric


@register_metric
class Rouge(PairwiseMetric):
    """
    Rouge evaluates rouge scores of produced hypotheses by comparing with references.

    Args:
        ngram: ['1', '2', 'l'] stands for ['rouge-1', 'rouge-2', 'rouge-l']
    """

    def __init__(self, lang='en', ngram='1,2,l', num_threads=1):
        super().__init__()
        self._num_threads = num_threads
        self._lang = lang
        self._ngram = ngram.split(',')
        self._rouge = RG()

    def eval(self):
        """
        Evaluate the performance with buffered hypotheses and references.
        """

        if self._score is not None:
            return self._score
        else:
            score = self._rouge.get_scores(self._hypos, self._refs, avg=True)
            self._score = {name: score[f'rouge-{name}']['f'] for name in self._ngram}
        return self._score
