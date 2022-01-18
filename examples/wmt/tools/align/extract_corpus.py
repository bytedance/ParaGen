import argparse
import json
import os
import re
from collections import defaultdict
from queue import PriorityQueue
from typing import List

from alignment import Alignment
from helper import _is_token_alnum
from tqdm import tqdm

COVER_RATE_THRESHOLD = 0.3


class Sentence:
    """
    Sentence object
    Attribute:
        - text: original text
        - lang: language code
        - intermediate: tokenizing result
        - quality: ratio of reserved chars and original chars. (excluding spaces)
    """

    def __init__(self, text: str, lang=None):
        self.text = re.sub(re.compile(r"[\s]+"), " ", text)

        self.lang = lang
        self._intermediate = str()

    @property
    def intermediate(self):
        return self.text

    @property
    def quality(self):
        return self._calculate_quality()

    def _calculate_quality(self):
        return len(self.intermediate.replace(" ", "")) / len(self.text.replace(" ", ""))


class Pair(object):
    def __init__(self, score, pair):
        self.score = score
        self.pair = pair

    def __lt__(self, other):
        return self.score < other.score

    def __gt__(self, other):
        return self.score > other.score

    def __eq__(self, other):
        return self.score == other.score


def extract_parallel_pairs(src_sentences_group: List[Sentence],
                           trg_sentences_group: List[Sentence],
                           aligner) -> List:
    """
    extract parallel sentence pairs from 2 separate sentence groups.
    :param src_sentences_group:
    :param trg_sentences_group:
    :param verbose:
    :return:
    """
    QUALITY_THRESHOLD = 0.6
    pairs = list()

    # use heap to store maximum src-trg pair
    trg_src_dict = defaultdict(PriorityQueue)
    src_trg_dict = defaultdict(PriorityQueue)

    threshold = 5  # default threshold
    for i, (src_sentence, trg_sentence) in tqdm(enumerate(zip(src_sentences_group, trg_sentences_group))):
        src_lang = src_sentence.lang
        trg_lang = trg_sentence.lang
        parallelism, _ = aligner.calculate(src_sentence.intermediate,
                                           trg_sentence.intermediate,
                                           src_lang,
                                           trg_lang,
                                           algorithm="default")

        if parallelism > threshold * 0.6 and \
                src_sentence.quality > QUALITY_THRESHOLD and \
                trg_sentence.quality > QUALITY_THRESHOLD:
            valid_src_tokens = {word for word in src_sentence.intermediate.split()
                                if _is_token_alnum(word)}
            valid_trg_tokens = {word for word in trg_sentence.intermediate.split()
                                if _is_token_alnum(word)}

            trg_src_dict[trg_sentence.text].put(
                Pair(-parallelism, (src_sentence, trg_sentence, len(valid_src_tokens), len(valid_trg_tokens))))
            src_trg_dict[src_sentence.text].put(
                Pair(-parallelism, (src_sentence, trg_sentence, len(valid_src_tokens), len(valid_trg_tokens))))

    for trg in trg_src_dict:
        candidate_pair = trg_src_dict[trg].get()
        score = - round(candidate_pair.score, 2)
        pair = candidate_pair.pair
        q = src_trg_dict.get(pair[0].text)
        if q:
            if q.get().pair[1].text == trg:
                if score >= threshold * 0.9 and \
                        pair[2] >= 5 and pair[3] >= 5 and \
                        aligner.clean_rules(pair[0].intermediate, pair[1].intermediate,
                                            pair[0].text, pair[1].text) == 0:
                    # filter with recall pairs
                    pairs.append((pair[0].text, pair[1].text, score))
    return pairs


def process(args, src_sentences_group, trg_sentences_group) -> List:
    """
    process each record.
    :param line:
    :return:
    """
    output_parallels = []
    # align sentence and extract parallel pairs
    if src_sentences_group and trg_sentences_group:
        group_sentences_min_cnt = min(len(src_sentences_group), len(trg_sentences_group))

        src_group = []
        trg_group = []
        for ss in src_sentences_group:
            sentence = Sentence(ss, src_lang)
            src_group.append(sentence)

        for ts in trg_sentences_group:
            sentence = Sentence(ts, trg_lang)
            trg_group.append(sentence)

        norm_lang_pair = '2'.join(sorted([src_lang, trg_lang]))  # zh2en -> en2zh
        aligner = Alignment(norm_lang_pair, args.wd)
        parallel_pairs = extract_parallel_pairs(src_group, trg_group, aligner)
        if parallel_pairs:
            rate = round(10.0 * len(parallel_pairs) / group_sentences_min_cnt, 2)
            output_parallels = [json.dumps({"src_text": pair[0],
                                            "trg_text": pair[1],
                                            "parallelism": pair[2],
                                            "src_lang": src_lang,
                                            "trg_lang": trg_lang,
                                            "rate": rate
                                            }, ensure_ascii=False)
                                for pair in parallel_pairs]
    return output_parallels


def main(args):
    src_sentences = []
    trg_sentences = []
    with open(os.path.join(args.wd, args.corpus_file1), 'r', encoding='utf-8') as f1:
        line1 = f1.readline()
        while line1:
            src_sentences.append(line1.strip())
            line1 = f1.readline()

    with open(os.path.join(args.wd, args.corpus_file2), 'r', encoding='utf-8') as f2:
        line2 = f2.readline()
        while line2:
            trg_sentences.append(line2.strip())
            line2 = f2.readline()

    parallel_data = process(args, src_sentences, trg_sentences)
    with open(os.path.join(args.wd, args.output_path), 'w', encoding='utf-8') as fout:
        if parallel_data:
            for data in parallel_data:
                fout.write(data + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mining parallel corpus from text. '
                                                 'python mining_corpus_single_page.py '
                                                 '-p en2zh,en2pt,en2nl')

    parser.add_argument('--lang_pair', '-p', type=str,
                        required=True,
                        help='Specify the source language.')
    parser.add_argument('--corpus_file1', '-f1', type=str,
                        required=True,
                        help='corpus file1 path.')
    parser.add_argument('--corpus_file2', '-f2', type=str,
                        required=True,
                        help='corpus file2 path.')
    parser.add_argument('--output_path', '-o', type=str,
                        required=True,
                        help='output file path.')
    parser.add_argument('--wd', '-wd', type=str,
                        default=".",
                        help='Specify the working directory, i.e. data')

    args = parser.parse_args()
    lang_pairs = tuple(args.lang_pair.split("2"))
    src_lang, trg_lang = lang_pairs
    main(args)
