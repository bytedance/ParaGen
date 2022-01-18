import argparse

import sacrebleu

parser = argparse.ArgumentParser()
parser.add_argument('--hypo_filename', metavar='N', type=str, help='hypo_filename')
parser.add_argument('--ref_filename', metavar='N', type=str, help='ref_filename')
parser.add_argument('--out_filename', metavar='N', type=str, help='out_filename')
args, unknown = parser.parse_known_args()

with open(args.hypo_filename, 'r') as fhypo, open(args.ref_filename, 'r') as fref, open(args.out_filename, 'w') as fout:
    max_bleu = 0
    for hypo, ref in zip(fhypo, fref):
        sent_bleu = sacrebleu.sentence_bleu(hypo, [ref]).score / 100
        fout.write(f'{sent_bleu}\n')
        max_bleu = max(max_bleu, sent_bleu)
