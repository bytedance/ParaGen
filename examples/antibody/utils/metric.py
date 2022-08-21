import sys
import json

import numpy as np 
from sklearn.metrics import roc_auc_score,matthews_corrcoef,precision_recall_fscore_support


def readJson(fname):
    data = []
    with open(fname) as fin:
        for line in fin:
            data.append(json.loads(line))
    print('Read {} examples from {}'.format(len(data), fname))
    return data

def calACC(hypo, gold):
    hypo = np.array(hypo)
    gold = np.array(gold)
    hypo[hypo>=0.5] = 1
    hypo[hypo<0.5] = 0
    correct = (hypo == gold).sum()
    total = len(gold)
    return correct / total

def calROC(hypo, gold):
    # print([(h, g) for h, g in zip(hypo, gold)])
    roc = roc_auc_score(np.array(gold), np.array(hypo))
    return roc

def calMCC(hypo, gold):
    hypo = np.array(hypo)
    gold = np.array(gold)
    hypo[hypo>=0.5] = 1
    hypo[hypo<0.5] = -1
    gold[gold==0] = -1
    mcc = matthews_corrcoef(gold, hypo)
    return mcc

def calF(hypo, gold, return_all=False):
    hypo = np.array(hypo)
    gold = np.array(gold)
    hypo[hypo>=0.5] = 1
    hypo[hypo<0.5] = 0
    p,r,f,_ = precision_recall_fscore_support(gold, hypo, average='binary')
    if return_all:
        return p, r, f
    return f


## python eval.py acc hypo gold

if __name__ == "__main__":
    mode = sys.argv[1]
    hypofile, goldfile = sys.argv[2], sys.argv[3]
    
    hypo = readJson(hypofile)
    gold = readJson(goldfile)
    gold = sum(gold, [])

    section = ["H1", "H2", "H3", "L1", "L2", "L3"]
    results = {k: {'pred':[], 'gold':[]} for k in section}
    for i, (h, g) in enumerate(zip(hypo, gold)):
        if 'section' in g.keys():
            sec = g['section']
        else:
            sec = section[i%6]
        seq, label = g['cdr_seq'], g['cdr_label']
        results[sec]['pred'].extend(h)
        results[sec]['gold'].extend(label)

    if mode != 'all':
        func = mode.upper()
        pred, gold = [], []
        for k, v in results.items():
            res = eval("cal{}(v['pred'], v['gold'])".format(func))
            print('{}\t{}'.format(k, res))
            pred.extend(v['pred'])
            gold.extend(v['gold'])
        tot = eval("cal{}(pred, gold)".format(func))
        print('Total\t{}'.format(tot))

    else:
        sections = []
        pred, gold = [], []
        for k, v in results.items():
            sections.append(k)
            pred.append(v['pred'])
            gold.append(v['gold'])

        func = ['ACC', 'MCC', 'ROC', 'F']
        
        for k,h,g in zip(section, pred, gold):
            for f in func:
                res = eval("cal{}(v['pred'], v['gold'])".format(f))
                print('{}\t{}\t{}'.format(k, f, res))

        pred = sum(pred,[])
        gold = sum(gold, [])
        for f in func:
            tot = eval("cal{}(pred, gold)".format(f))
            print('Total\t{}\t{}'.format(f, tot))

