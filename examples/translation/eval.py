import argparse
import sacrebleu

from mosestokenizer import MosesTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--gold', metavar='N', type=str, help='refence path')
parser.add_argument('--hypo', metavar='N', type=str, help='hypothesis path')

args, unknown = parser.parse_known_args()
gold_path = args.gold
hypo_path = args.hypo

tok = MosesTokenizer(lang='de')
with open(gold_path, 'r') as fin1, open(hypo_path, 'r') as fin2:
    with open(f'{gold_path}.tok', 'w') as fo1, open(f'{hypo_path}.tok', 'w') as fo2:
        golds, hypos = [], []
        for g, h in zip(fin1, fin2):
            g, h = ' '.join(tok(g)), ' '.join(tok(h))
            golds.append(g)
            hypos.append(h)
            fo1.write(f'{g}\n')
            fo2.write(f'{h}\n')

    scores = sacrebleu.corpus_bleu(hypos, [golds], force=True, tokenize='none')
    print(f'Tokenized BLEU {scores.score}')
