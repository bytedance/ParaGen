import argparse

import sacrebleu

parser = argparse.ArgumentParser()
parser.add_argument('--hypo_filename', metavar='N', type=str, help='hypo_filename')
parser.add_argument('--out_filename', metavar='N', type=str, help='out_filename')
parser.add_argument('--num_candidates', type=int, help="num_candidates")
args, unknown = parser.parse_known_args()

with open(args.hypo_filename, 'r') as fhypo, open(args.out_filename, 'w') as fout:
    max_bleu = 0
    buffer = []
    for i, hypo in enumerate(fhypo):
        buffer.append(hypo)
        if i % args.num_candidates == args.num_candidates - 1:
            for i, h in enumerate(buffer):
                bleu = sacrebleu.sentence_bleu(h, [r for r in buffer if r != h])
                fout.write(f'{bleu.score / 100}\n')
            buffer.clear()
