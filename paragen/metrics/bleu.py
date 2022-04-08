import sacrebleu

from paragen.metrics import PairwiseMetric, register_metric


@register_metric
class BLEU(PairwiseMetric):
    """
    BLEU evaluates BLEU scores of produced hypotheses by comparing with references.
    """

    def __init__(self, no_tok=False, lang='en', **kwargs):
        super().__init__()
        self._no_tok = no_tok
        self._lang = lang

        self._sacrebleu_kwargs = kwargs
        if self._no_tok:
            self._sacrebleu_kwargs['tokenize'] = 'none'
        else:
            self._sacrebleu_kwargs['tokenize'] = get_tokenize_by_lang(self._lang)

    def build(self, *args, **kwargs):
        """
        Build metric
        """
        self.reset()

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
            refs = list(zip(*self._refs))
            bleu = sacrebleu.corpus_bleu(self._hypos, refs, **self._sacrebleu_kwargs)
            self._score = bleu.score
        return self._score


def get_tokenize_by_lang(lang):
    if lang in ['zh']:
        return 'zh'
    elif lang in ['ko']:
        return 'char'
    else:
        return '13a'
