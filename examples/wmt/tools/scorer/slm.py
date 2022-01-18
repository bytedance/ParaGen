# Before running this command, you should firstly run:
# pip install kenlm
import argparse

import kenlm

parser = argparse.ArgumentParser()
parser.add_argument('--hypo_filename', metavar='N', type=str, help='hypo_filename')
parser.add_argument('--out_filename', metavar='N', type=str, help='out_filename')
args, unknown = parser.parse_known_args()

model_name = "your_model_name.arpa"

model = kenlm.LanguageModel(model_name)
with open(args.hypo_filename, 'r') as fhypo, open(args.out_filename, 'w') as fout:
    for hypo in fhypo:
        hypo = hypo.strip("\n").replace('<unk>', ' ')
        fout.write(f'{model.score(hypo) / (len(hypo.split()) + 1e-2)}\n')
