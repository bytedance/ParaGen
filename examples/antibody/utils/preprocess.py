import os
import json
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

random.seed(888)
np.random.seed(888)

############################## io ##############################

def read_txt(path):
    data = []
    with open(path) as fin:
        for line in fin:
            data.append(line.strip())
    print('Read {} examples from {}'.format(len(data), path))
    return data

def read_fasta(path):
    data = {}
    with open(path) as fin:
        name = None
        for line in fin:
            if line.startswith('>'):
                name = line.strip().lstrip('>')
            else:
                seq = line.strip()
                data[name] = seq
    print('Read {} examples from {}'.format(len(data), path))
    return data

def read_json(path):
    data = []
    with open(path) as fin:
        for line in fin:
            data.append(json.loads(line))
    print('Read {} examples from {}'.format(len(data), path))
    return data

def save_json(data, path):
    with open(path, 'w') as fout:
        for d in data:
            fout.write('{}\n'.format(json.dumps(d)))
    print('Save {} examples to {}'.format(len(data), path))


############################## disease ##############################

# SARS={"Disease": "SARS-COV-2", "Vaccine": "None","Chain": "Heavy", "Unique sequences":10000, "Isotype": "IGHG", "Species": "human", "BSource": "PBMC"}
Health={"Disease": "None", "Vaccine": "None","Chain": "Heavy", "Unique sequences":10000, "Isotype": "IGHG", "Species": "human", "BSource": "PBMC"}
FilterCond={"Vaccine": "None","Chain": "Heavy", "Unique sequences":10000, "Isotype": "IGHG", "Species": "human", "BSource": "PBMC"}

def disease(disease, savename):
    # handle.disease('SARS-COV-2', 'disease.sars.jsonl')
    info = read_json('data/OAS_label.jsonl')

    conds = FilterCond
    conds['Disease'] = disease
    selected = []
    for l in info:
        l.pop('Link')
        flag = True
        for k, v in conds.items():
            if flag == False:
                break
            if k == 'Unique sequences':
                if l['Unique sequences'] <= v:
                    print(k, l[k], v)
                    flag = False
            else:
                if l[k] != v:
                    print(k, l[k], v)
                    flag = False
        
        if flag:
            selected.append(l)
    print(len(selected), len(set([json.dumps(x) for x in selected])))
    save_json(selected, savename)

def select_sequence(filename, savename, top=100):
    # handle.select_sequence('disease.sars.jsonl', 'disease.sars.top100.jsonl')
    rootdir = '../germline_aligned'
    data = read_json(filename)

    sequences = []
    for d in data:
        basename = '{}_{}_{}.txt'.format(d['Run'], d['Chain'], d['Isotype'])
        path = os.path.join(rootdir, basename)
        profile = read_json(path)
        for ins in profile:
            ins.update(d)
        print(len(profile), len(set([x['sequence'] for x in profile])))
        profile = sorted(profile, key=lambda x:x['redundency'], reverse=True)
        sequences.extend(profile[:top])
    save_json(sequences, savename)



############################## cross validation ##############################

def cross_split(filename, savename, k=10):
    # handle.cross_split('cell.jsonl', 'cell_germ/cross')
    data = []
    with open(filename) as fin:
        for line in fin:
            data.append(json.loads(line))
    print('Read {} example from {}'.format(len(data), filename))

    savedir = '/'.join(savename.split('/')[:-1])
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    num = k
    for i in range(num):
        random.shuffle(data)
        train, valid = data[:int(len(data) * 0.9)], data[int(len(data) * 0.9):]
        save_json(train, '{}_train_{}_all.json'.format(savename, i))
        save_json(valid, '{}_valid_{}_all.json'.format(savename, i))

# based on individual
def cross_split_for_disease(filename, savename, k=10):
    # handle.cross_split_for_disease('sars.top100.jsonl', 'sars_germ_top100/cross')
    data = []
    with open(filename) as fin:
        for line in fin:
            data.append(json.loads(line))
    print('Read {} example from {}'.format(len(data), filename))

    savedir = '/'.join(savename.split('/')[:-1])
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    # runs = list(set([x['Run'] for x in data]))
    health_runs = list(set([x['Run'] for x in data if x['Disease']=="None"]))
    disease_runs = list(set([x['Run'] for x in data if x['Disease']!="None"]))
    health_runs = health_runs[:len(disease_runs)]
    runs = disease_runs + health_runs

    # try five times
    for i in range(5):
        contflag = False
        random.shuffle(data)
        random.shuffle(runs)

        num = k
        chunk_size = len(runs) // k
        print('Chunk {}, size {}'.format(k, chunk_size))
        for i in range(num):        
            valid_runs = runs[i * chunk_size : (i+1) * chunk_size]
            train_runs = [x for x in runs if x not in valid_runs]
            print('Valid Runs {}:{}'.format(i * chunk_size, (i+1) * chunk_size))
            print('Train Runs {}'.format(len(train_runs)))
            valid = [x for x in data if x['Run'] in valid_runs]
            train = [x for x in data if x['Run'] in train_runs]
            save_json(train, '{}_train_{}_all.json'.format(savename, i))
            save_json(valid, '{}_valid_{}_all.json'.format(savename, i))

            # ensure that validation set contains both two categories
            diseaseRun = [x for x in valid if x['Disease']!="None"]
            healthRun = [x for x in valid if x['Disease']=="None"]
            print(len(diseaseRun), len(healthRun))
            if len(diseaseRun) == 0 or len(healthRun) == 0:
                contflag = True
                break

        if contflag != True:
            break



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

def searchCDR(data):
    cdr_db = "dataset/Antibody/sars_cdr3"
    run_template = '{run}_Heavy_IGHG.csv.gz.txt'

    prev_run_file = ""
    prev_run_info = None
    for d in data:
        run_file = os.path.join(cdr_db, run_template.format(run=d['Run']))
        if run_file == prev_run_file:
            run_info = prev_run_info
        else:
            run_info = read_json(run_file)
            prev_run_info = run_info
            prev_run_file = run_file
        seq2cdr = {x['sequence_alignment_aa']: x for x in run_info}
        seq = d['sequence'].replace('-', '')
        # print(seq)
        cdr_info = seq2cdr.get(seq, None)
        if cdr_info:
            d.update(cdr_info)
    return data

def UpdateCDR(args):
    # handle.UpdateCDR('sars.germ.jsonl', 'sars.germ.cdr.jsonl')
    inname, outname = args.i, args.o
    data = read_json(inname)
    ndata = searchCDR(data)
    save_json(ndata, outname)


