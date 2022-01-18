from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--input", action="store")
parser.add_argument("--output", action="store")
parser.add_argument('--maxlen', metavar='N', type=int, default=510)
parser.add_argument('--doc', action="store_true")

args = parser.parse_args()


fin = open(args.input)
fout = open(args.output, "w")

last_doc = []
for line in fin:
    words = line.strip().split(" ")
    if not args.doc:
        words = last_doc + words
    while len(words) > args.maxlen:
        cur, rest = words[:args.maxlen], words[args.maxlen:]
        fout.write(" ".join(cur) + "\n")
        words = rest
    if len(words) != 0:
        if args.doc:
            fout.write(" ".join(words) + "\n")
        else:
            last_doc = words

fout.close()
