from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json
import logging

import cld3
from tqdm import tqdm

logger = logging.getLogger(__name__)


def main(fin, fout, probability, proportion):
    output = open(fout, "w")
    for line in tqdm(open(fin)):
        line_object = json.loads(line)
        src_text, trg_text = line_object['src_text'], line_object['trg_text']
        src_lang, trg_lang = line_object['src_lang'], line_object['trg_lang']
        mono = line_object['monolingual']

        is_src_reliable = False
        is_trg_reliable = False     
        is_src_rev_reliable = False
        is_trg_rev_reliable = False

        if src_lang is not None and trg_lang is not None:
            if len(src_text.split(" ")) <= 3 and len(trg_text.split(" ")) <= 3:
                is_src_reliable, is_trg_reliable = True, True
            else:
                ld_src_list = cld3.get_frequent_languages(src_text, num_langs=3)
                ld_trg_list = cld3.get_frequent_languages(trg_text, num_langs=3)
                for ld_src in ld_src_list:
                    if ld_src.language == src_lang and ld_src.probability >= probability and ld_src.proportion >= proportion:
                        is_src_reliable = True
                    elif ld_src.language == trg_lang and ld_src.probability >= probability and ld_src.proportion >= proportion:
                        is_src_rev_reliable = True
                if mono:
                    is_trg_reliable = True
                    is_trg_rev_reliable = False
                else:
                    for ld_trg in ld_trg_list:
                        if ld_trg.language == trg_lang and ld_trg.probability >= probability and ld_trg.proportion >= proportion:
                            is_trg_reliable = True
                        elif ld_trg.language == src_lang and ld_trg.probability >= probability and ld_trg.proportion >= proportion:
                            is_trg_rev_reliable = True
                # tell whether to reverse
                if is_src_reliable is False and is_trg_reliable is False:
                    if is_src_rev_reliable and is_trg_rev_reliable:
                        src_text, trg_text = trg_text, src_text
                        is_src_reliable, is_trg_reliable = True, True
                        # sys.stderr.write(f">>> Reverse a pair: {json.dumps(line_object)}\n")
                        line_object['src_text'], line_object['trg_text'] = src_text, trg_text

        if is_src_reliable and is_trg_reliable:
            output.write(json.dumps(line_object) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='filter sentences in batch according to language')
    parser.add_argument("--fin", type=str)
    parser.add_argument("--fout", type=str)
    parser.add_argument('--probability', type=float,
                        default=0.8,
                        help='Specify the probability threshold for cld3 detection, range: 0 ~ 1.')

    parser.add_argument('--proportion', type=float,
                        default=0.6,
                        help='Specify the proportion threshold for cld3 detection, range: 0 ~ 1.')

    args = parser.parse_args()
    main(fin=args.fin, fout=args.fout, probability=args.probability, proportion=args.proportion)
