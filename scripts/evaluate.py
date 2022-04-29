import argparse

from paragen.metrics import create_metric
from paragen.utils.data import possible_eval

parser = argparse.ArgumentParser()
parser.add_argument('--hypo', type=str, help='hypothesis')
parser.add_argument('--ref', default='', type=str, help='ground truth reference')
parser.add_argument('--metric', default=None, type=str, help='metric class name')
args, unknown = parser.parse_known_args()

kwargs = {}
current_key = None
for ele in unknown:
    if ele.startswith("--"):
        current_key = ele[2:]
    elif current_key is not None:
        kwargs[current_key] = possible_eval(ele)
        current_key = None

metric = create_metric(args.metric, **kwargs)
metric.build()

with open(args.hypo, 'r ') as fhypo, open(args.ref, 'r') as fref:
    for hypo, ref in zip(fhypo, fref):
        hypo, ref = hypo.strip('\n'), ref.strip('\n')
        metric.add(hypo, ref)

scores = metric.eval()
print(scores)
