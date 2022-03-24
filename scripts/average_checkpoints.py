import argparse
import os
from typing import List

import torch

from paragen.utils.tensor import get_avg_ckpt

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, help='directory path')
parser.add_argument('--prefix', default='', type=str, help='prefix split with comma')
parser.add_argument('--output', default=None, type=str, help='output path')
args, unknown = parser.parse_known_args()


def startswith_exists(filename: str, valid_prefix: List[str]):
    for p in valid_prefix:
        if filename.startswith(p):
            return True
    return False


prefix = args.prefix.split(',')
ckpt_list = []
for f in os.listdir(args.dir):
    if startswith_exists(f, prefix):
        ckpt_list.append(f'{args.dir}/{f}')

avg_ckpt = get_avg_ckpt(ckpt_list)
output_path = args.output or f'{args.dir}/{args.prefix}_avg.pt'
with open(output_path, 'wb') as fout:
    torch.save(avg_ckpt, fout)


