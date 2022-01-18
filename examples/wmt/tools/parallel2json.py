import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--src_file", type=str)
parser.add_argument("--trg_file", type=str)
parser.add_argument("--src_lang", type=str)
parser.add_argument("--trg_lang", type=str)
parser.add_argument("--output_file", type=str)

args = parser.parse_args()
print(args)

src_filename = args.src_file
trg_filename = args.trg_file

src_basename = os.path.basename(src_filename)

out_filename = args.output_file
src_lang = args.src_lang
trg_lang = args.trg_lang

if trg_lang is None:
    trg_lang = ""

if trg_filename is not None:
    with open(src_filename, "r") as src_file, open(trg_filename, "r") as trg_file, open(out_filename, "w") as outfile:
        for src_line, trg_line in zip(src_file, trg_file):
            data = {
                "src_text": src_line.strip(),
                "trg_text": trg_line.strip(),
                "src_lang": src_lang,
                "trg_lang": trg_lang,
                "data_source": src_basename,
                "monolingual": False, "pseudo": False, "preprocessed": False
            }
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
else:
    with open(src_filename, "r") as src_file, open(out_filename, "w") as outfile:
        for src_line in src_file:
            data = {
                "src_text": src_line.strip(),
                "trg_text": "",
                "src_lang": src_lang,
                "trg_lang": trg_lang,
                "data_source": src_basename,
                "monolingual": True, "pseudo": False, "preprocessed": False
            }
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
