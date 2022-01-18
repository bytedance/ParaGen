import os
from transformers import MBart50TokenizerFast
from tqdm import tqdm
from paragen.tokenizers import create_tokenizer
from typing import List
import numpy as np


def checkout(path='raw', outpath='preprocessed'):
    skip_list = set(["cs_CZ-en_XX", "af_ZA-en_XX", "iu_CA-en_XX"])
    lang_pairs = set()
    for file in os.listdir(path):
        prefix, lang_pair, lang = file.rsplit('.', 2)
        lang_pairs.add(lang_pair)
    print(f"Skip_list: {skip_list}")
    for i in lang_pairs:
        if i in skip_list:
            continue
        for prefix in ["train", "valid", "test"]:
            src, tgt = i.split('-')
            print(f"Process: {prefix}.{i}")
            try:
                length1 = len(open(f"{path}/{prefix}.{i}.{src}", encoding='utf-8').readlines())
                length2 = len(open(f"{outpath}/{prefix}.index.{i}.{src}", encoding='utf-8').readlines())
                if length1 != length2:
                    print(f"{prefix}.{i}, {length1}, {length2}")
            except:
                pass


def make_tokenize(f_src, f_tgt, f_src_out, f_tgt_out, src_lang, tgt_lang, skip=0, use_hug=True, max_length=256):

    if use_hug:
        tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang=src_lang, tgt_lang=tgt_lang)
        eos = 2
    else:
        tokenizer = create_tokenizer({"class": "FastBPE",
                                      "vocab": "vocab_spm_100/vocab-spm-100k.dict",
                                      "preserved_tokens": "vocab_spm_100/preserved_tokens.txt"})
        tokenizer.build()
        eos = tokenizer.eos_token

    batch_size = 10000
    num = skip
    now = 0
    while now < skip:
        f_src.readline()
        if f_tgt: 
            f_tgt.readline()
        now += 1
    now = 0

    while True:
        tmp_src_texts = []
        tmp_tgt_texts = []
        while now < batch_size:
            src_text = f_src.readline()
            if f_tgt: 
                tgt_text = f_tgt.readline()

            if src_text:
                tmp_src_texts.append(' '.join(src_text.split()[:max_length + 2]))
                if f_tgt: 
                    tmp_tgt_texts.append(' '.join(tgt_text.split()[:max_length + 2]))
                now += 1
            else:
                break
        num += now
        now = 0
        if len(tmp_src_texts) == 0:
            break

        print(f"{src_lang}-{tgt_lang}: {num}")

        if use_hug:
            src_ids = tokenizer(tmp_src_texts, return_tensors="pt", padding=True).input_ids[:, 1:max_length + 2].tolist()
            if f_tgt: 
                with tokenizer.as_target_tokenizer():
                    tgt_ids = tokenizer(tmp_tgt_texts, return_tensors="pt", padding=True).input_ids[:, 1:max_length + 2].tolist()
        else:
            src_ids = tokenizer.token2index(tmp_src_texts)
            if f_tgt: 
                tgt_ids = tokenizer.token2index(tmp_tgt_texts)

        
        for i, line in enumerate(src_ids):
            line += [eos]
            line = line[:line.index(eos)]
            f_src_out.write(f"{tmp_src_texts[i]}\t{line}" + '\n')
        
        if f_tgt: 
            for i, line in enumerate(tgt_ids):
                line += [eos]
                line = line[:line.index(eos)]
                f_tgt_out.write(f"{tmp_tgt_texts[i]}\t{line}" + '\n')

    print("Complete")


def main(path='raw', outpath='preprocessed', mono=False, use_hug=False, max_length=256):
    skip_list = set(["cs_CZ-en_XX", "af_ZA-en_XX", "iu_CA-en_XX"])
    lang_pairs = set()
    for file in os.listdir(path):
        prefix, lang_pair, lang = file.rsplit('.', 2)
        lang_pairs.add(lang_pair)
    print(f"Skip_list: {skip_list}")
    for i in lang_pairs:
        if 'tr' not in i:
            continue
        if i in skip_list:
            continue
        for prefix in ["train", "valid", "test"]:
            src, tgt = i.split('-')
            print(f"Process: {prefix}.{i}")
            
            f_src = None
            f_tgt = None
            f_src_out = None
            f_tgt_out = None

            try:
                skip = 0
                if os.path.exists(f"{outpath}/{prefix}.index.{i}.{src}"):
                    with open(f"{outpath}/{prefix}.index.{i}.{src}", encoding='utf-8') as f:
                        skip = len(f.readlines())
                f_src = open(f"{path}/{prefix}.{i}.{src}", encoding='utf-8')
                f_src_out = open(f"{outpath}/{prefix}.index.{i}.{src}", "a+", encoding='utf-8')
                if not mono:
                    f_tgt = open(f"{path}/{prefix}.{i}.{tgt}", encoding='utf-8')
                    f_tgt_out = open(f"{outpath}/{prefix}.index.{i}.{tgt}", "a+", encoding='utf-8') 
                make_tokenize(f_src, f_tgt, f_src_out, f_tgt_out, src, tgt, skip, use_hug=use_hug, max_length=max_length)
            except:
                pass
            finally:
                if f_src: f_src.close()
                if f_tgt: f_tgt.close()
                if f_src_out: f_src_out.close()
                if f_tgt_out: f_tgt_out.close()

            
if __name__ == "__main__":
    main(path="raw", outpath="preprocessed")
