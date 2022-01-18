# Before running this command, you should firstly run:
# pip install fairseq
# pip install fastBPE
# wget https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.en.tar.gz
# tar zxvf wmt19.en.tar.gz
import argparse
from itertools import islice

import numpy as np
from fairseq.models.transformer_lm import TransformerLanguageModel

parser = argparse.ArgumentParser()
parser.add_argument('--hypo_filename', metavar='N', type=str, help='hypo_filename')
parser.add_argument('--out_filename', metavar='N', type=str, help='out_filename')
# parser.add_argument('--num_candidates', type=int, help="num_candidates")
args, unknown = parser.parse_known_args()

en_lm = TransformerLanguageModel.from_pretrained('wmt19.en', 'model.pt', tokenizer='moses', bpe='fastbpe')
en_lm.cuda()

num_processed = 0
ppl = []
batch_num = 1000
with open(args.hypo_filename, 'r') as f, open(args.out_filename, 'w') as out:
    while True:
        n_lines = list(map(lambda x: x.strip(), islice(f, batch_num)))
        if len(n_lines) == 0:
            break
        for ele in en_lm.score(n_lines, beam=1):
            ppl.append(float(ele['positional_scores'].mean().neg().exp().item()))
        num_processed += batch_num
        print(f"Processed {num_processed}")

    ppl = np.array(ppl)
    ppl = np.nan_to_num(ppl, nan=np.nanmax(ppl))
    # scores = 1 - ppl/ppl.max()
    # for ele in zip(ppl.tolist(), scores.tolist()):
    #     out.write(f"{np.log(ele[0])}, {ele[0]}, {ele[1]}\n")

    ppl = np.array(ppl)
    for ele in ppl.tolist():
        out.write(f"{np.log(ele)}\n")
