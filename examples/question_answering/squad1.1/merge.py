import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--hypo', metavar='N', type=str, help='hypo Path')
parser.add_argument('--id', metavar='N', type=str, help='index Path')
parser.add_argument('--out', metavar='N', type=str, help='output path')

def main(hypo_path, idx_path, output_path):
    output = {}
    with open(idx_path, 'r') as fin1, open(hypo_path, 'r') as fin2:
        for i, hypo in zip(fin1, fin2):
            i, hypo = i.strip('\n'), hypo.strip('\n')
            output[i] = hypo
    with open(f'{output_path}', 'w') as fout:
        json.dump(output, fout, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.hypo, args.id, args.out)
