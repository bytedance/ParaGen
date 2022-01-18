import logging
import math
import os
from functools import lru_cache

from dbdict import DbDict
from helper import NUMCHAR_ONLY, REPEAT_CHAR, REPEAT_SYMBOL, _is_token_alnum, unique_seq

logger = logging.getLogger(__name__)


class Alignment(object):
    def __init__(self, lang_pair, wd=None):
        self.lang_pair = lang_pair
        self.dict_name = '-'.join(sorted(lang_pair.split('2'))) + '.db'
        if wd is not None:
            self.dict_name = os.path.join(wd, '-'.join(sorted(lang_pair.split('2'))) + '.db')
        if os.path.exists(self.dict_name):
            self.dict_mapping = DbDict(self.dict_name, True)
        else:
            raise Exception("Please check the path of the dict.")

    @lru_cache(maxsize=256 * 1024)
    def get_key(self, key):
        """
        wrapper for lookup single vocabulary
        :param key:
        :return:
        """
        return self.dict_mapping.get(key)

    @lru_cache(maxsize=16 * 1024)
    def get_keys(self, keys):
        """
        wrapper for batch lookup
        :param keys:
        :return:
        """
        return self.dict_mapping.mget(keys)

    def word_translation_prob(self, src_word, trg_word, threshold=0.05):
        """
        Alternative method to calculate cross lingual word similarity.
        Replace with any cosine sim of available word embedding.
        :param src_word:
        :param trg_word:
        :param threshold:
        :return:
        """
        try:
            prob = self.get_key(src_word)[trg_word]
            return 1 if prob >= threshold else prob
        except Exception:
            return 0

    def lookup_vocabs(self, direction, words):
        lst_words = [f'{direction}{word}' for word in words]

        if len(lst_words) <= 5:
            pw_vocabs_tuple = [(pw, self.get_key(pw)) for pw in lst_words]
        else:
            # src_trg_words_tuple = get_keys(tuple(prefix_ls_words))
            pw_vocabs_tuple = self.get_keys(tuple(lst_words))

            # reorder with src words
            pw_vocabs_tuple = sorted(pw_vocabs_tuple,
                                     key=lambda item: lst_words.index(item[0]))
        return pw_vocabs_tuple

    def calculate(self, ls, lt,
                  src_lang, trg_lang,
                  lowercase=True, debug=False, algorithm="default"):
        debug_info = dict()
        s2t = f"{src_lang}2{trg_lang}"
        t2s = f"{trg_lang}2{src_lang}"

        if lowercase:
            ls = ls.lower()
            lt = lt.lower()

        # as set
        ls_words = unique_seq(word for word in ls.split() if _is_token_alnum(word))
        lt_words = unique_seq(word for word in lt.split() if _is_token_alnum(word))

        ls_words_dict = {ls_word: 0 for ls_word in ls_words}
        lt_words_dict = {lt_word: 0 for lt_word in lt_words}

        total_words = len(ls_words) + len(lt_words)
        if len(ls_words) == 0 or len(lt_words) == 0:
            word_ratio = 0
        else:
            word_ratio = min(len(ls_words) / len(lt_words), len(lt_words) / len(ls_words))
            if word_ratio > 0.4:
                word_ratio = 1

        # penalty for token number ratio, ideal case with same tokens number, and bp = 1
        bp = round(math.exp(word_ratio - 1), 3)

        # length confidence is used to filter out short sentence
        # 24 chars of sentence could result in 0.9 .  7 chars result in 0.5 .
        try:
            length_confidence = 1 - (math.exp(-0.1 * len(ls.encode())) +
                                     math.exp(-0.1 * len(lt.encode()))) / 2
        except Exception as ex:
            length_confidence = 0.0
            logger.exception(ex)

        if total_words == 0:
            total_words += 1

        src2trg_mapping_cnt = 0

        src_trg_words_tuple = self.lookup_vocabs(s2t, ls_words)

        for prefix_ls_word, trg_words in src_trg_words_tuple:
            if trg_words:
                for g_trg_word in trg_words:
                    if g_trg_word in lt_words_dict:
                        src2trg_mapping_cnt += 1
                        lt_words_dict.pop(g_trg_word, 0)
                        break
        if debug:
            debug_info['src2trg_overlap'] = src2trg_mapping_cnt
            debug_info['total_words'] = total_words

        src2trg_overlap = src2trg_mapping_cnt / max(len(ls_words), 1)

        # because lt_words is changed
        trg2src_mapping_cnt = 0
        trg_src_words_tuple = self.lookup_vocabs(t2s, lt_words)

        for prefix_lt_word, src_words in trg_src_words_tuple:
            if src_words:
                for g_src_word in src_words:
                    if g_src_word in ls_words_dict:
                        trg2src_mapping_cnt += 1
                        ls_words_dict.pop(g_src_word)
                        break
        if debug:
            debug_info['trg2src_overlap'] = trg2src_mapping_cnt
            debug_info['penalty'] = bp
        trg2src_overlap = trg2src_mapping_cnt / max(len(lt_words), 1)

        score = 2 / (1 / max(0.1, src2trg_overlap) + 1 / max(0.1, trg2src_overlap))
        overall_score = round(bp * length_confidence * 10.0 * score, 2)
        return overall_score, debug_info

    @staticmethod
    def clean_rules(ls, lt, origin_ls, origin_lt):
        """

        :param ls: tokenized source text
        :param lt: tokenized target text
        :param origin_ls: original source text
        :param origin_lt: original target text
        :return:
        """
        # drop very long sentences
        if len(origin_ls) > 2500 or len(origin_lt) > 2500:
            return 1

        ls = ls.lower()
        lt = lt.lower()

        ls_words_list = [word for word in ls.split() if _is_token_alnum(word)]
        lt_words_list = [word for word in lt.split() if _is_token_alnum(word)]

        ls_words = set(ls_words_list)
        lt_words = set(lt_words_list)

        # empty token
        if len(ls_words) < 1 or len(lt_words) < 1:
            return 4

        # token number ratio
        if len(ls_words) / len(lt_words) >= 4 or len(lt_words) / len(ls_words) >= 4:
            return 5

        # duplicate tokens ratio filter
        if len(ls_words_list) / len(ls_words) > 2 or len(lt_words_list) / len(lt_words) > 2:
            return 6

        # repeat symbols or chars
        origin_ls_rm_space = "".join(origin_ls.split())
        origin_lt_rm_space = "".join(origin_lt.split())

        r1 = REPEAT_CHAR.search(origin_ls_rm_space)
        r2 = REPEAT_CHAR.search(origin_lt_rm_space)
        r3 = REPEAT_SYMBOL.search(origin_ls_rm_space)
        r4 = REPEAT_SYMBOL.search(origin_lt_rm_space)
        if r1 or r2 or r3 or r4:
            if r1 and r2 and r1.group(0) == r2.group(0):
                return 0
            if r3 and r4 and r3.group(0) == r4.group(0):
                return 0
            return 8
        r5 = NUMCHAR_ONLY.search(origin_ls_rm_space)
        r6 = NUMCHAR_ONLY.search(origin_lt_rm_space)
        if r5 or r6:
            return 12
        return 0
