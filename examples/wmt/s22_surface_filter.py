import argparse
import json
import logging
import re
import sys
from string import punctuation

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

console = logging.StreamHandler()
logger.addHandler(console)

NUMCHAR_ONLY = re.compile(r"^[\d\W ]+$")

REPEAT_SYMBOL = re.compile(r'\W{6,}')  # ?$#@^&# or !!!!!!
REPEAT_CHAR = re.compile(r'(?P<char>\w{1,4})(?P=char){6,}')  # 的的的的的 hahahahaha
NUMCHAR_ONLY = re.compile(r"^[\d\W ]+$")


# ---- duplicate filter --- #

def dedup_repeated_pair(src_sent, tgt_sent):
    # faster using bash
    raise NotImplementedError


def repeated_tokens_ratio_filter(sent, ratio=2):
    sent_list = sent.strip().split()
    unique_words = set(sent_list)

    if len(sent_list) / (len(unique_words) + 1e-7) > ratio:
        return False

    return True


def repeated_char_filter(sent):
    sent_rm_space = "".join(sent.strip().split())
    r = REPEAT_CHAR.search(sent_rm_space)

    if r:
        return False
    return True


def repeated_symbol_filter(sent):
    sent_rm_space = "".join(sent.strip().split())
    r1 = REPEAT_SYMBOL.search(sent_rm_space)

    if r1:
        return True
    return True


# --- html filter --- #
def html_tag_filter(sent):
    detector = re.compile('<.*?>')
    html_tag = re.findall(detector, sent)
    if html_tag or 'https://' in sent or 'http://' in sent:
        return False

    return True


# --- length filter --- #
def word_len_filter(sent, max_sgl_word_len=40, max_avg_word_len=20, min_avg_word_len=2):
    """
    condition: too long words or too long/short average words
    """

    sent_list = sent.strip().split()
    word_lens = [len(word) for word in sent_list]
    avg_word_len = sum(word_lens) / (len(sent_list) + 1e-7)
    if max(word_lens) > max_sgl_word_len or avg_word_len > max_avg_word_len or avg_word_len < min_avg_word_len:
        return False

    return True


def sent_len_filter(src_sent, tgt_sent, max_sent_len=250, min_sent_len=3, ratio=3):
    # should use bash

    def _single_sent_len_filter(sent):
        sent_len = len(sent.strip().split())
        return max_sent_len > sent_len > min_sent_len

    def _ratio_sent_len_filter(src, tgt):
        src_sent_len = len(src.strip().split())
        tgt_sent_len = len(tgt.strip().split())

        return src_sent_len / (tgt_sent_len + 1e-9) < ratio and tgt_sent_len / (src_sent_len + 1e-9) < ratio

    if tgt_sent is not None:
        return _single_sent_len_filter(src_sent) and _single_sent_len_filter(tgt_sent) and _ratio_sent_len_filter(
            src_sent, tgt_sent)
    else:
        return _single_sent_len_filter(src_sent)


# ---punc filter --- #
def numchar_only_filter(sent):
    src_sent_rm_space = "".join(sent.strip().split())

    r = NUMCHAR_ONLY.search(src_sent_rm_space)

    if r:
        return False
    return True


def punc_filter(sent, abs_num=10, in_ratio=0.5):
    punc_set = set(punctuation)
    sent = sent.strip()
    cnt_func = lambda l1, l2: sum([1 for x in l1 if x in l2])

    num_punc = cnt_func(sent, punc_set)
    str_len = len(sent)

    if num_punc < abs_num and num_punc / (str_len + 1e-9) < in_ratio:
        return True

    return False


def num_ratio_filter(sent, ratio=0.5):
    if len(re.findall("[\d\-\|/]", sent)) / (len(sent) + 1e-7) > ratio:
        return False

    return True


# --specific char filter -- #
def hard2print_char_filter(sent):
    if r"\x" in sent:
        return False
    return True


# data format:
# {"src_text": "这是源句子",
# "trg_text": "This is target sentence",
# "src_lang": "zh",
# "tgt_lang": "en",
# "monolingual": false,
# "pseudo": false,
# "preprocessed": false,
# "category": 0
# }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--monolingual", action="store_true")

    args = parser.parse_args()
    # pprint(args)

    mono_sent_filter_func_list = [repeated_tokens_ratio_filter, repeated_char_filter, html_tag_filter, word_len_filter,
                                  punc_filter, num_ratio_filter, hard2print_char_filter]
    parallel_sent_filter_func_list = [sent_len_filter]

    for line in sys.stdin:
        if not line:
            continue
        try:
            line_object = json.loads(line)
            src_text = line_object['src_text']
            trg_text = None if args.monolingual else line_object['trg_text']
            trg_text = None if trg_text == "" else trg_text  # treat empty string as None

            success = True
            filtered_func = None
            for func in parallel_sent_filter_func_list:
                if not func(src_text, trg_text):
                    success = False
                    filtered_func = func.__name__
                    break
            if success:
                for func in mono_sent_filter_func_list:
                    if not success:
                        break
                    for sent in [src_text, trg_text]:
                        if not success:
                            break
                        if sent is not None:
                            sent = sent.strip()
                        else:
                            break
                        if not func(sent):
                            success = False
                            filtered_func = func.__name__

            line_object['is_filtered'] = not success
            line_object['filtered_func'] = filtered_func
            line = json.dumps(line_object)

            sys.stdout.write(f"{line.strip()}\n")
        except Exception as ex:
            logger.exception(ex)


if __name__ == '__main__':
    main()
