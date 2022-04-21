from functools import reduce

from rouge import Rouge as RG
from torch.multiprocessing import Queue, Process

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
            if self._num_threads > 1:
                size = len(self.hypos)
                part_num = size // self._num_threads + 1
                queue, pthreads, scores = Queue(), [], []
                for i in range(self._num_threads):
                    sidx, eidx = i * part_num, min((i + 1) * part_num, size)
                    p = Process(target=eval_func,
                                args=(queue, self._rouge, self.hypos[sidx:eidx], self.refs[sidx:eidx], i))
                    p.start()
                    pthreads.append(p)
                for p in pthreads:
                    scores.append(queue.get())
                    p.join()
                scores = reduce(lambda x, y: x + y, scores)

                tot_scores = {
                    f'rouge-{name}': 0.
                    for name in self._ngram
                }
                for s in scores:
                    for name in self._ngram:
                        tot_scores[f'rouge-{name}'] += s[f'rouge-{name}']['f']
                self._score = {name: tot_scores[f'rouge-{name}'] / size for name in self._ngram}
            else:
                score = self._rouge.get_scores(self._hypos, self._refs, avg=True)
                self._score = {name: score[f'rouge-{name}']['f'] for name in self._ngram}
        return self._score


def eval_func(queue: Queue, metric, hypos, refs, i):
    queue.put(metric.get_scores(hypos, refs, avg=False))
