from rouge_metric import PerlRouge
from rouge_metric import PyRouge

from paragen.metrics import PairwiseMetric, register_metric


@register_metric
class Rouge(PairwiseMetric):
    """
    Rouge evaluates rouge scores of produced hypotheses by comparing with references.

    Args:
        ngram: ['1', '2', 'l'] stands for ['rouge-1', 'rouge-2', 'rouge-l']
    """

    def __init__(self, lang='en', ngram='1,2,l'):
        super().__init__()
        self._lang = lang
        self._ngram = ngram.split(',')
        self._rouge = PerlRouge(
            rouge_n_max=3, rouge_l=True, rouge_w=True, rouge_w_weight=1.2,
            rouge_s=True, rouge_su=True, skip_gap=4
        ) if lang == 'en' else PyRouge(
            rouge_n=(1, 2, 4), rouge_l=True, rouge_w=True,
            rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4
        )

    def add(self, hypo, ref):
        """
        Add parallel hypotheses and references to metric buffer
        """
        if isinstance(ref, str):
            ref = [ref]
        self._hypos.append(hypo)
        self._refs.append(ref)

    def eval(self):
        """
        Evaluate the performance with buffered hypotheses and references.
        """
        if self._score is not None:
            return self._score
        else:
            score = self._rouge.evaluate(self._hypos, self._refs)
            self._score = {name: score[f'rouge-{name}']['f'] for name in self._ngram}
        return self._score
