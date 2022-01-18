# layer drop

layerdrop can speed up training and meanwhile improve performance.
model trained by layerdrop is robust to layer pruning at inference as well.

training: each layer is randomly dropped with probability p
inference: with full network.

# args in layerdrop_config:
prob: [float] initial probability of dropping layers
gamma: [float] prob decay in PLD (DeepSpeed)
mode_depth: [str] 'transformer' or 'bert', the ways of prob distribution across layers

# performance 

''' WMT14-ende
| Model | best BLEU | Training Speed Up
|Transformer-big (baseline) | 27.12| ---
| + layerdrop (p=0.2) | 27.74 | 11%
| + layerdrop (p=0.3, mode='transformer', gamma=1e-4) | 27.56 | 8%
| + layerdrop (p=0.3, mode='transformer') | 27.80 | 8%

''' WMT14-enfr
| Model | best BLEU | Training Speed Up
|Transformer-big (baseline) | 39.35| ---
| + layerdrop (p=0.3, mode='transformer') | 39.85 | 8%

# Training Command

First prepare vocab and dataset in folder 'data', then run:

'''
bash run.sh -lib examples/structured_dropout/strucdrop -config examples/structured_dropout/configs/train.yaml
'''

# head and ffn drop

Similar to layer drop, but the dropped target is number of head in MHA and FFN dim.

change args ‘attn_structured_dropout’ and ‘ffn_structured_dropout’ to adjust probability.
