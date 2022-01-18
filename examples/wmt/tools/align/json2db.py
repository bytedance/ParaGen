import argparse
import json
import logging
import os

from dbdict import DbDict
from helper import _is_token_alnum
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_dict(wd, src_trg_pair):
    """
    load dict file to memory
    :param src_trg_pair en2zh, en2fr, ar2en ...
    :return:
    """

    src_lang, trg_lang = sorted(src_trg_pair.split("2"))
    s2t = f"{src_lang}2{trg_lang}"
    t2s = f"{trg_lang}2{src_lang}"

    # establish db
    db_path = os.path.join(wd, src_lang + '-' + trg_lang + '.db')
    dict_mapping = DbDict(db_path, True)

    src2trg_dict_name = f'dict.{src_lang}2{trg_lang}'
    trg2src_dict_name = f'dict.{trg_lang}2{src_lang}'

    try:
        # s2t
        s2t_dict_path = os.path.join(wd, src2trg_dict_name)
        d = dict()
        with open(s2t_dict_path) as f:
            for item in tqdm(f):
                dictionary = json.loads(item.strip())
                for k, v in dictionary.items():
                    if _is_token_alnum(k):
                        d[f'{s2t}{k}'] = v
            dict_mapping.update(d)

        # t2s
        t2s_dict_path = os.path.join(wd, trg2src_dict_name)
        d = dict()
        with open(t2s_dict_path) as f:
            for item in tqdm(f):
                dictionary = json.loads(item.strip())
                for k, v in dictionary.items():
                    if _is_token_alnum(k):
                        d[f'{t2s}{k}'] = v
            dict_mapping.update(d)

    except Exception as ex:
        logger.exception(ex)
        exit(-1)
    logger.info(f"Dictionary loaded for {s2t}/{t2s}")

    return dict_mapping


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='scoring sentences in batch'
                                                 '-pairs ja2zh -method idf -score 0 -threshold 0.6')
    parser.add_argument('--lang_pair', '-pair', type=str,
                        required=True,
                        help='Specify the language pair, i.e. en2zh')
    parser.add_argument('--wd', '-wd', type=str,
                        default=".",
                        help='Specify the working directory, i.e. data')
    args = parser.parse_args()
    dict_mapping = load_dict(args.wd, args.lang_pair)
