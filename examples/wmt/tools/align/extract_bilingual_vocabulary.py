import argparse
import json
import os
from collections import Counter, defaultdict

from helper import _is_token_alnum

THRESHOLD = 0.01
GAP = 10


def get_full_mapping(src_filename, trg_filename, align_filename,
                     mapping_filename, reverse_src2trg=False, lowercase=True):
    """ Get full mapping give align.

    Args:
        src_filename:
        trg_filename:
        align_filename:
        mapping_filename:
        reverse_src2trg:
        lowercase:

    Returns:

    """
    print('src: {}, trg: {}, align: {}, mapping: {}, reverse: {}'.format(
        src_filename, trg_filename, align_filename, mapping_filename,
        reverse_src2trg))

    src2trg_mapping = defaultdict(lambda: defaultdict(int))

    processed_line = 0
    with open(src_filename) as fs, open(trg_filename) as ft, open(
            align_filename) as fa:
        for ls, lt, la in zip(fs, ft, fa):
            if lowercase:
                ls = ls.lower()
                lt = lt.lower()
            processed_line += 1
            ls_words = ls.split()
            lt_words = lt.split()
            la_aligns = la.split()

            src_pos_counter = Counter()
            trg_pos_counter = Counter()
            valid_src_pos = set()
            valid_trg_pos = set()
            for align in la_aligns:
                # only consider one-to-one mapping
                src_pos, trg_pos = align.split('-')
                src_pos = int(src_pos)
                trg_pos = int(trg_pos)
                # only consider alpha number token
                if _is_token_alnum(ls_words[src_pos]):
                    src_pos_counter[src_pos] += 1
                if _is_token_alnum(lt_words[trg_pos]):
                    trg_pos_counter[trg_pos] += 1

            # ignore token that aligned twice
            for pos, c in src_pos_counter.items():
                if c == 1:
                    valid_src_pos.add(pos)
            for pos, c in trg_pos_counter.items():
                if c == 1:
                    valid_trg_pos.add(pos)

            for align in la_aligns:
                src_pos, trg_pos = align.split('-')
                src_pos = int(src_pos)
                trg_pos = int(trg_pos)
                if _is_token_alnum(ls_words[src_pos]) and _is_token_alnum(
                        lt_words[trg_pos]) and (src_pos in valid_src_pos) and (
                        trg_pos in valid_trg_pos):
                    if reverse_src2trg:
                        src2trg_mapping[lt_words[trg_pos]][
                            ls_words[src_pos]] += 1
                    else:
                        src2trg_mapping[ls_words[src_pos]][
                            lt_words[trg_pos]] += 1

            if processed_line % 1000000 == 0:
                print('{} done.'.format(processed_line))

    with open(mapping_filename, 'w') as fw:
        print('dump to {} ...'.format(mapping_filename))
        json.dump(src2trg_mapping, fw)

    return src2trg_mapping


def refine_dict(full_mapping, clean_dict_filename, threshold, ignore_gap):
    """ Clean dictionary based on frequency and gap of frequency.
    For example,
    {'s1': ['t1': 999, 't2': 199, 't3':1],
     's2': ['m1': 2000, 'm2': 100]}
     =>
    {'s1': ['t1': 999, 't2': 199],
     's2': ['m1': 2000]}

    Args:
        full_mapping:
        clean_dict_filename:
        threshold:
        ignore_gap:

    Returns:

    """
    print('Refine dict to {}, threshold: {}, ignore_gap: {} ...'.format(
        clean_dict_filename, threshold, ignore_gap))
    full_mapping = sorted(
        full_mapping.items(),
        key=lambda x: sum(x[1].values()),
        reverse=True)

    with open(clean_dict_filename, 'w') as fw:
        for idx, src2trg in enumerate(full_mapping):
            src = src2trg[0]
            trg = sorted(src2trg[1].items(), key=lambda x: x[1], reverse=True)
            total_count = sum(c[1] for c in trg)
            clean_trg = dict()
            p = trg[0][1]
            for w, c in trg:
                if c / total_count < threshold:
                    # too rare
                    break
                if (p / c > ignore_gap) and (c / total_count < THRESHOLD * 5):
                    # large gap
                    break
                p = c
                clean_trg.update({w: round(c / total_count, 3)})

            fw.write('{}\n'.format(json.dumps({src: clean_trg}, ensure_ascii=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process alignments and do filter')
    parser.add_argument('--src_filename',
                        help='Origin src file name before bsp',
                        type=str,
                        required=True)
    parser.add_argument('--trg_filename',
                        help='Origin trg file name before bsp',
                        type=str,
                        required=True)
    parser.add_argument('--align_filename',
                        help='align file name by atools',
                        type=str,
                        required=True)
    parser.add_argument('--dict_filename',
                        help='clean dict file name',
                        type=str,
                        required=True)
    parser.add_argument('--threshold',
                        help='threshold of ignore frequency',
                        type=float,
                        default=THRESHOLD)
    parser.add_argument('--ignore_gap',
                        help='gap of ignore frequency',
                        type=float,
                        default=GAP)
    parser.add_argument(
        '--overwrite', dest='overwrite',
        action='store_true', help='Overwrite existing output files')
    args = parser.parse_args()
    if args.overwrite:
        print('Overwrite existing file')

    src2trg_mapping_filename = '{}.{}'.format(args.align_filename,
                                              'src2trg_mapping')
    trg2src_mapping_filename = '{}.{}'.format(args.align_filename,
                                              'trg2src_mapping')
    if os.path.isfile(src2trg_mapping_filename) and (not args.overwrite):
        print('loading mapping: {}'.format(src2trg_mapping_filename))
        with open(src2trg_mapping_filename) as f:
            full_src2trg_mapping = json.load(f)
    else:
        print('creating mapping: {}'.format(src2trg_mapping_filename))
        full_src2trg_mapping = get_full_mapping(args.src_filename,
                                                args.trg_filename,
                                                args.align_filename,
                                                src2trg_mapping_filename,
                                                False)

    if os.path.isfile(trg2src_mapping_filename) and (not args.overwrite):
        print('loading mapping: {}'.format(trg2src_mapping_filename))
        with open(trg2src_mapping_filename) as f:
            full_trg2src_mapping = json.load(f)
    else:
        print('creating mapping: {}'.format(trg2src_mapping_filename))
        full_trg2src_mapping = get_full_mapping(args.src_filename,
                                                args.trg_filename,
                                                args.align_filename,
                                                trg2src_mapping_filename,
                                                True)

    src2trg_clean_dict_filename = '{}.{}'.format(args.dict_filename,
                                                 'src2trg')
    refine_dict(full_src2trg_mapping, src2trg_clean_dict_filename,
                args.threshold, args.ignore_gap)

    trg2src_clean_dict_filename = '{}.{}'.format(args.dict_filename,
                                                 'trg2src')
    refine_dict(full_trg2src_mapping, trg2src_clean_dict_filename,
                args.threshold, args.ignore_gap)
