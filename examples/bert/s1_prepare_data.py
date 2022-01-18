import argparse
from random import randint
import sys
from multiprocessing import Pool
import datasets
import time
import re
import os

# per-batch = 256*512 ~ 128000

# bert: after tokenization
# wiki should be 2500M
# book should be 800M
# total: 3300M
# 40 epoch = 100M / (3300M / 128000)

# roberta: 
# total: 4000M
# 32 epoch = 100M / (4000M / 128000)
# wiki should be 3030M
# book should be 969M

# 49836008  3273586621 14832780171 wiki.bpe
# 23581325  1417162160  6249128143 book.bpe

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=80)
    parser.add_argument("--data", default='valid')
    args = parser.parse_args()

    if args.data in [ 'wiki', 'valid' ]:
        inputs = datasets.load_dataset("wikipedia", "20200501.en")['train']
    elif args.data == 'book':
        inputs = datasets.load_dataset("bookcorpusopen")['train']
    else:
        raise Exception
    # inputs = itertools.islice(inputs, 0, 100)

    if args.data == 'valid':
        tmp_inputs = []
        for i in range(3000):
            x = randint(0, len(inputs))
            tmp_inputs.append(inputs[x])
        inputs = tmp_inputs


    worker = MPWorker(args)
    pool = Pool(args.num_workers)
    results = pool.imap(worker.process, inputs, 1000)
    out = open(f"{args.data}.raw-0", 'w')

    start = time.time()
    for i, batch in enumerate(results):
        if batch[0] == 'P' and len(batch[1]) > 0:
            if len(batch[1]) > 0:
                for ele in batch[1]:
                    out.write(ele + '\n')
                out.write('\n')
        if i % 10000 == 0:
            end = time.time()
            print("processed {} lines, total cost {}s".format(
                i, int(end - start)),
                  file=sys.stderr)
    out.close()


def clean_wiki(text: str, verbose= False):
    ret = []
    section = text.split("\n\n")
    for sec in section:
        # remove some sections
        if sec.startswith("References") \
            or sec.startswith("External links") \
                or sec.startswith("Category"):
            if verbose:
                print("--> REMOVE", sec)
            continue
        paras = sec.split("\n")
        # remove headers of a section or short sentences
        for para in paras:
            para = para.strip()
            if para.count(" ") > 10:
                ret.append(para)
            else:
                if verbose:
                    print("--> REMOVE", para)
    return ret


def clean_book(text: str, verbose= False):
    ret = []
    lines = re.split("\n+", text)
    for line in lines:
        line = line.strip()
        if line.count(" ") > 10:
            ret.append(line)
        else:
            if verbose:
                print("--> REMOVE", line)
    return ret


class MPWorker(object):
    def __init__(self, args):
        self.args = args

    def process(self, doc):
        try:
            if self.args.data == 'wiki':
                content = clean_wiki(doc['text'])
            else:
                content = clean_book(doc['text'])
            ret = ['P', content]
        except Exception as e:
            print(e, 'error occues when processing', doc['text'][:50])
            ret = ['F', None]
        return ret


if __name__ == "__main__":
    main()
