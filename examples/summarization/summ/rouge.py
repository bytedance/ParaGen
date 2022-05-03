import time
import logging
logger = logging.getLogger(__name__)

import nltk
from pyrouge import Rouge155
from torch.multiprocessing import Queue, Process

from paragen.metrics import PairwiseMetric, register_metric
from paragen.utils.io import mkdir, remove_tree, UniIO


@register_metric
class Rouge(PairwiseMetric):
    """
    Rouge evaluates rouge scores of produced hypotheses by comparing with references.
    Thanks Chenxin An for his helpful suggestion on rouge reproduction.

    Args:
        ngram: ['1', '2', 'l'] stands for ['rouge-1', 'rouge-2', 'rouge-l']
    """

    def __init__(self, lang='en', ngram='1,2,l', num_threads=1):
        super().__init__()
        self._num_threads = num_threads
        self._lang = lang
        self._ngram = ngram.split(',')

    def add(self, hypo, ref):
        if hypo == '':
            hypo = '<empty>'
        else:
            hypo = '\n'.join(nltk.sent_tokenize(hypo))
            ref = '\n'.join(nltk.sent_tokenize(ref))
        super().add(hypo, ref)

    def eval(self):
        """
        Evaluate the performance with buffered hypotheses and references.
        """

        if self._score is not None:
            return self._score
        else:
            size = len(self.hypos)
            if self._num_threads > 1:
                part_num = size // self._num_threads + 1
                queue, pthreads, outputs = Queue(), [], []
                for i in range(self._num_threads):
                    sidx, eidx = i * part_num, min((i + 1) * part_num, size)
                    p = Process(target=eval_func,
                                args=(queue, self.hypos[sidx:eidx], self.refs[sidx:eidx], i))
                    p.start()
                    pthreads.append(p)
                for p in pthreads:
                    outputs.append(queue.get())
                    p.join()

                scores = {}
                for key in outputs[0][0]:
                    tot_scores, cnt = 0, 0
                    for s, c in outputs:
                        tot_scores += s[key] * c
                        cnt += c
                    scores[key] = tot_scores / cnt
            else:
                scores, _ = eval(self._hypos, self._refs)

            scores = {
                f'rouge-{name}': scores[f'rouge_{name}_f_score']
                for name in self._ngram
            }
            self._score = {name: scores[f'rouge-{name}'] for name in self._ngram}
        return self._score


def eval_func(queue: Queue, hypos, refs, idx):
    output = eval(hypos, refs, tmp_dir=f'.tmp_pyrouge/{idx}')
    queue.put(output)


def eval(hypo, ref, tmp_dir='.tmp_pyrouge'):
    cnt = len(hypo)
    current_time = str(time.time()).replace('.', '')

    tmp_dir = f'{tmp_dir}/{current_time}'
    hypo_dir, ref_dir = f'{tmp_dir}/hypo', f'{tmp_dir}/ref'
    mkdir(tmp_dir)
    mkdir(hypo_dir), mkdir(ref_dir)

    def write(url, s):
        with UniIO(url, 'w', encoding='utf-8') as f:
            f.write(s)

    for i in range(cnt):
        if len(ref[i]) < 1:
            continue
        write(f'{hypo_dir}/hypo.{i}.txt', hypo[i])
        write(f'{ref_dir}/ref.{i}.txt', ref[i])

    r = Rouge155()
    r.log.setLevel(logging.WARN)
    r.model_dir, r.system_dir = ref_dir, hypo_dir
    r.model_filename_pattern, r.system_filename_pattern = 'ref.#ID#.txt', r'hypo.(\d+).txt',
    rouge_results = r.convert_and_evaluate()

    results_dict = r.output_to_dict(rouge_results)

    remove_tree(tmp_dir)
    return results_dict, cnt
