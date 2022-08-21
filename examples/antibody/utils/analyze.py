import os
import sys
import json
import random
import argparse

import pandas as pd
import numpy as np

from .metric import calACC, calF, calMCC, calROC

random.seed(666)

############################## io ##############################

def readJson(name):
    data = []
    with open(name) as fin:
        for line in fin:
            data.append(json.loads(line))
    print('Read {} samples from {}'.format(len(data), name))
    return data

def readLog(name):
    data = []
    with open(name) as fin:
        for line in fin:
            data.append(line.strip())
    print('Read {} lines from {}'.format(len(data), name))
    return data

def saveJson(data, name):
    with open(name, 'w') as fout:
        for d in data:
            fout.write('{}\n'.format(json.dumps(d)))
            fout.flush()
    print('Save {} examples to {}'.format(len(data), name))

def saveLog(data, name):
    with open(name, 'w') as fout:
        for d in data:
            fout.write('{}\n'.format(d))
            fout.flush()
    print('Save {} examples to {}'.format(len(data), name))


############################## utils ##############################

def GetPatientLevel(hypos, refers):
    patients = {}
    for r,h in zip(refers, hypos):
        run = r['Run']
        if run  not in patients:
            patients[run] = {'Disease': r['Disease'], 'hypos': []}
        patients[run]['hypos'].append(float(h))

    for k,v in patients.items():
        hypos = v['hypos']
        sorted(hypos)
        number = len(hypos)
        trimmed_hypos = hypos[int(number*0.1):-int(number*0.1)]
        trimmed_mean = sum(trimmed_hypos) / len(trimmed_hypos)
        v['prob'] = trimmed_mean

    labels, preds = [], []
    for k,v in patients.items():
        if v['Disease'] == 'None':
            labels.append(0)
        else:
            labels.append(1)
        preds.append(v['prob'])
    acc, mcc, roc, f = calACC(preds, labels), calMCC(preds, labels), calROC(preds, labels), calF(preds, labels)
    print('Acc {}\tMCC {}\tROC {}\tF {}'.format(acc, mcc, roc, f))
    states = {'acc':acc, 'mcc': mcc, 'roc': roc, 'f':f}
    return patients, states

############################## mode ##############################


def GetCrossResult(args):
    # python3 examples/antibody/utils/analyze.py -mode GetCrossResult -i test.log -flag 1
    k = args.k
    if args.flag == 1:
        show_flag = ['best.valid.acc', 'best.valid.mcc', 'best.valid.roc', 'best.valid.f', 'best.valid.precision', 'best.valid.recall']
    elif args.flag == 0:
        show_flag = ['best.valid.pearson','best.valid.spearman','best.valid.r2']
    else:
        show_flag = ['best.valid.acc']
    logs = readLog(args.i)
    results = []
    head = True
    for l, c in enumerate(logs):
        if head and c.endswith('Create environment with'):
            head = False
            continue
        if c.endswith('Create environment with') or c.endswith('Killing all processes...'):
            n = 1
            while 'best.valid' not in logs[l-n]:
                n += 1
            results.append(logs[l-n])
    if len(results) == k - 1:
        n = 1
        while 'best.valid' not in logs[-n]:
            n += 1
        results.append(logs[-n])
    assert len(results) == k, '{}-cross but get {} results'.format(k, len(results))

    organized = []
    for res in results:
        subset = {}
        for ex in res.split(' | '):
            if ex.startswith('best'):
                k, v = ex.split(' ')
                subset[k] = float(v)
        organized.append(subset)

    print('Subset\t' + '\t'.join(show_flag))
    select_scores = []
    for i, score in enumerate(organized):
        select_score = [score[k] for k in show_flag]
        print('cross-{}\t{}'.format(i, '\t'.join([str(round(x,4)) for x in select_score])))
        select_scores.append(select_score)
    select_scores = np.array(select_scores)
    print('Avg\t{}'.format('\t'.join([str(round(x,4)) for x in select_scores.mean(axis=0)])))
    print('Std\t{}'.format('\t'.join([str(round(x,4)) for x in select_scores.std(axis=0)])))    \

def GetBestIndividual(args):
    # python3 examples/antibody/analyze.py -mode GetBestIndividual \
    #               -i ${WOKESHOP}/${MODEL}_${NAME}/${MODEL}_${NAME}_${i} \
    #               -d ${DATADIR}/${DATA}/cross_valid_${i}_all.json 
    refers = readJson(args.d)
    if os.path.isdir(args.i):
        hypo_files = sorted([os.path.join(args.i, f) for f in os.listdir(args.i)])
    else:
        hypo_files = [args.i]

    metric_name = args.assess_by
    best_states, best_epoch = {metric_name: -1}, 0
    best_patient = None
    for i, hfile in enumerate(hypo_files):
        hypos = readLog(hfile)
        patients, states = GetPatientLevel(hypos, refers)
        if states[metric_name] > best_states[metric_name]:
            best_epoch = int(hfile.split('/')[-1].lstrip('valid_epoch').rstrip('.hypo'))
            best_states = states
            best_patient = patients

    print('#'*100)
    print('Epoch {} | {} \n'.format(best_epoch, json.dumps(best_states)))
    print('Run\tDisease\tProb')
    for k,v in best_patient.items():
        print('{}\t{}\t{}'.format(k, v['Disease'], v['prob']))
    print('\n\n')

def GetCrossIndividual(args):
    # python3 examples/antibody/analyze.py -mode GetCrossIndividual -i logs/${MODEL}_${NAME}_patient.log
    data = readLog(args.i)
    states = [json.loads(line.split('|')[-1].strip()) for line in data if line.startswith('Epoch')]

    show_flag = states[0].keys()
    print('Subset\t' + '\t'.join(show_flag))
    select_scores = []
    for i, score in enumerate(states):
        select_score = [score[k] for k in show_flag]
        print('cross-{}\t{}'.format(i, '\t'.join([str(round(x,4)) for x in select_score])))
        select_scores.append(select_score)
    select_scores = np.array(select_scores)
    print('Avg\t{}'.format('\t'.join([str(round(x,4)) for x in select_scores.mean(axis=0)])))
    print('Std\t{}'.format('\t'.join([str(round(x,4)) for x in select_scores.std(axis=0)])))    

def GetProbSeq(args):
    # python3 examples/antibody/analyze.py 
    #           -mode GetProbSeq 
    #           -i clusterRes_07_train_chunk_catpos_align_bz200_mutation_sars0513_2 -d sars_germ_sm_2
    #           -threshold_mode high 
    #           -threshold 0.8
    #           -m 0
    refer_file="data/{d}/cross_valid_{c}_all.json"
    hypo_file="results/{m}/{m}_{c}/valid_epoch{e}.hypo"
    patient_file="logs/{m}_patient.log".format(m=args.i)
    patient = readLog(patient_file)
    epoch = [int(line.split()[1]) for line in patient if line.startswith('Epoch')]
    print('Cross best epoch {}'.format(epoch))

    k=args.k
    patients = {}
    for cross in range(k):
        refer = readJson(refer_file.format(d=args.d, c=cross))
        xx = [x['sequence'] for x in refer]
        print('Cross {}: sequence {}, uniq sequence {}'.format(k, len(xx), len(set(xx))))
        pred = readLog(hypo_file.format(m=args.i, c=cross, e=epoch[cross]))
        for r,h in zip(refer, pred):
            run = r['Run']
            if run not in patients:
                patients[run] = {'Disease': r['Disease'], 'sequences': []}
            r['pred'] = float(h)
            patients[run]['sequences'].append(r)
    print('All run {}, uniq run {}'.format(len(patients.keys()), len(set(patients.keys()))))

    thre = args.threshold
    thre_mode = args.threshold_mode
    highprob = {}
    for k, v in patients.items():
        # print('{}\t{}'.format(k, v['Disease']))
        if v['Disease'] != 'None':
            details = v['sequences']
            for x in details:
                if x['sequence'] not in highprob:
                    highprob[x['sequence']] = []
                highprob[x['sequence']].append(x['pred'])
        
    reduce_highprob = [{'sequence':k, 'pred': sum(v)/len(v)} for k,v in highprob.items()]
    reduce_highprob = sorted(reduce_highprob, key=lambda x:x['pred'], reverse=True)
    if thre_mode == 'high':
        predition = [[p] for p in reduce_highprob if p['pred'] >= thre]
    else:
        predition = [[p] for p in reduce_highprob if p['pred'] < thre]
               
    print('Meet condition [{}-{}]: sequence {}'.format(thre_mode, thre, len(predition)))

    saveJson(predition, patient_file.replace('patient', '{}prob{}_uniq'.format(thre_mode, str(thre).replace('.', ''))))

############################## high prob disease seq ##############################

def caldist(x, y, mode='distance'):
    import Levenshtein as L
    if mode == 'distance':
        dist_func = L.distance
    elif mode == 'hamming':
        dist_func = L.hamming
    else:
        raise NotImplementedError
    return dist_func(x, y)

def MatchCDR(cdr, db, thre=0.7):
    for x in db:
        dist = caldist(cdr, x)
        ratio = 1 -  dist / len(cdr)
        if ratio >= thre:
            # print(cdr, x, dist, ratio)
            return x, ratio
    return None, None


def FindNewBinder(args):
    # python3 examples/antibody/analyze.py -mode FindNewBinder 
    #           -i eatlm_sars.log
    #           -threshold 0.85
    #           -o sars.highprob08
    savename = args.o
    thre = args.threshold

    highprob_file = "logs/{}".format(args.i)
    sars_db = "data/CoV-AbDab_Heavy_cdr3.csv"
    cdr_file = "data/disease.sars.germline.cdr.jsonl"

    data = readJson(highprob_file)
    uniqdata = [x[0] for x in data]

    cdr_info_data = readJson(cdr_file)
    cdr_dict = {x['sequence']:x for x in cdr_info_data}

    sars = pd.read_csv(sars_db)
    sars = sars.to_dict('list')
    sars_dict = {cdr: seq for seq, cdr in zip(sars['VH '], sars['CDRH3'])}

    found = []
    found_cnt = 0
    for i, d in enumerate(uniqdata):
        cdr_info = cdr_dict.get(d['sequence'], None)
        if cdr_info:
            cdr = cdr_info['cdr3_aa']
            found_cdr, found_similar = MatchCDR(cdr, sars_dict.keys(), thre=thre)
            if found_cdr:
                detail=d
                detail.update(cdr_info)
                detail['sars_db_seq'] = {'cdr':found_cdr, 'similarity':found_similar, 'seq': sars_dict[found_cdr]}
                found.append(detail)
                found_cnt += 1
        # print(i, found_cnt)
    sorted_found = sorted(found, key=lambda x:x['sars_db_seq']['similarity'], reverse=True)
    saveJson(found, '{}.uniq.jsonl'.format(savename))
    saveJson(sorted_found, '{}.sorted.uniq.jsonl'.format(savename))
    df = pd.DataFrame(found)
    df.to_csv('{}.uniq.csv'.format(savename))
    df = pd.DataFrame(sorted_found)
    df.to_csv('{}.sorted.uniq.csv'.format(savename))

def CumulateRankBinder(args):
    # python3 examples/antibody/utils/analyze.py -mode CumulateRankBinder -i . -o model.cdrmatch.json -threshold 0.85
    inputdir, savename = args.i, args.o
    thre = args.threshold
    files = [os.path.join(inputdir, x) for x in os.listdir(inputdir) if x.endswith('.log')]

    cdr_file = "data/disease.sars.germline.top100.cdr.jsonl"
    cdr_file_2 = "data/disease.health.germline.top100.cdr.jsonl"
    cdr_info_data = readJson(cdr_file)
    cdr_dict = {x['sequence']:x for x in cdr_info_data}
    cdr_info_data_2 = readJson(cdr_file_2)
    cdr_dict.update({x['sequence']:x for x in cdr_info_data_2})

    sars_db = "data/CoV-AbDab_Heavy_cdr3.csv"
    sars = pd.read_csv(sars_db)
    sars = sars.to_dict('list')
    sars_dict = {cdr: seq for seq, cdr in zip(sars['VH '], sars['CDRH3'])}

    flags = {}
    for highprob_file in files:
        data = readJson(highprob_file)
        uniqdata = [x[0] for x in data]
        if 'positive' in savename:
            uniqdata = [x for x in uniqdata if x['pred']>0.5]
            print('Positive data {}'.format(len(uniqdata)))

        found_flag = []
        for i, d in enumerate(uniqdata):
            cdr_info = cdr_dict.get(d['sequence'], None)
            if cdr_info:
                cdr = cdr_info['cdr3_aa']
                found_cdr, _ = MatchCDR(cdr, sars_dict.keys(), thre=thre)
                if found_cdr:
                    found_flag.append(1)
                else:
                    found_flag.append(0)
            else:
                print("Missing for {}".format(d['sequence']))
        print(sum(found_flag))
        basename = highprob_file.split('/')[-1]
        flags[basename] = found_flag
    # saveJson(flags, savename)
    json.dump(flags, open(savename, 'w'))
    



############################## Main function ##############################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default='EvalGermline')
    parser.add_argument('-i', default='result.log', help="input path (eg. log file)")
    parser.add_argument('-d', default='reference.json', help="dataset path")
    parser.add_argument('-o', default='output', help="output path")
    parser.add_argument("-flag", type=int, default=1, help="1 for binary classification; 2 for regression task; 0 for multiple classification")
    parser.add_argument("-k", type=int, default=10, help="k-cross validation")
    parser.add_argument("-threshold", type=float, default=0.5, help="threshold for new sars binder")
    parser.add_argument("-threshold_mode", type=str, default='high', help="high or low")
    parser.add_argument("-assess_by", type=str, default='f', help="metric to select best patient")

    args = parser.parse_args()

    eval('{}(args)'.format(args.mode))
