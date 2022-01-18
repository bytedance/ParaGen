# -*- coding: utf-8 -*-
# create@ 2020-02-29 00:26

from __future__ import absolute_import, division, print_function, unicode_literals

import re
import sys
import unicodedata

# Regular expression for clean use
REPEAT_SYMBOL = re.compile(r'\W{10,}')  # ?$#@^&# or !!!!!!
REPEAT_CHAR = re.compile(r'(?P<char>\w{1,4})(?P=char){10,}')  # 的的的的的 hahahahaha
NUMCHAR_ONLY = re.compile(r"^[\d\W ]+$")
DIGITS = re.compile("\d+")

# charset for filtering tokens
_ALPHANUMERIC_CHAR_SET = set(
    chr(i) for i in range(sys.maxunicode)
    if (unicodedata.category(chr(i)).startswith('L') or
        unicodedata.category(chr(i)).startswith('N')))
NON_BREAK_HYPHEN = chr(9602)  # '▂'

# This set contains all letter (L) & number (N) of unicode chars.
# concatenate words (in _ALPHANUMERIC_CHAR_SET) with NON_BREAK_HYPHEN
# should not be split
_ALPHANUMERIC_CHAR_SET.add(NON_BREAK_HYPHEN)


def _is_token_alnum(t):
    token_is_alnum = len(t) > 0 and (t[0] in _ALPHANUMERIC_CHAR_SET)
    return token_is_alnum


def unique_seq(seq):
    """
    Deduplicate sequence and reserve elements order.
    :param seq:
    :return:
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
