import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--input_filename", type=str)
parser.add_argument("--src_filename", type=str)
parser.add_argument("--trg_filename", type=str)

args = parser.parse_args()
print(args)

fin = open(args.input_filename)

src_out = open(args.src_filename, "w")
trg_out = open(args.trg_filename, "w")
for line in fin:
    dct = json.loads(line)
    src_out.write(f"{dct['src_text']}\n")
    trg_out.write(f"{dct['trg_text']}\n")

src_out.close()
trg_out.close()
