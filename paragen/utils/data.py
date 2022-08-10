from typing import Dict, List, Tuple
import json
import random

import numpy as np

from paragen.utils.io import SPACE_NORMALIZER


def split_tgt_sequence(tgt, bos, eos):
    """
    Split gold target into previous tokens and prediction target.
    For examples in text, `[hello, world, !] -> [<bos>, hello, world, !], [hello, world, !, <eos>]`

    Args:
        tgt: target sequence
        bos: begin-of-sequence index
        eos: end-of-sequence index

    Returns:
        - previous tokens
        - prediction target
    """
    if len(tgt[0]) > 0 and tgt[0][0] == bos and tgt[0][-1] == eos:
        prev_tokens = [v[:-1] for v in tgt]
        tgt = [v[1:] for v in tgt]
    else:
        prev_tokens = [[bos] + v for v in tgt]
        tgt = [v + [eos] for v in tgt]
    return tgt, prev_tokens


def reorganize(samples: List[Dict]):
    """
    Transforming List[Dict] to Dict[List] by grouping with keys

    Args:
        - samples: a list of samples
    """
    samples_ = {key: [] for key in samples[0]}
    for sample in samples:
        for key, val in sample.items():
            samples_[key].append(val)
    return samples_


def count_sample_token(sample):
    """
    Count sample tokens

    Args:
        sample: a piece of samples

    Returns:
        - total token numbers
    """
    if isinstance(sample, str):
        return len(SPACE_NORMALIZER.split(sample))
    elif isinstance(sample, list):
        return sum([count_sample_token(s) for s in sample])
    elif isinstance(sample, Dict):
        return sum([count_sample_token(s) for s in sample.values()])
    else:
        return 1


def transform_data(key, data):
    """
    Transform data

    Args:
        key:
        data:

    Returns:

    """
    if isinstance(data[0], Dict):
        return transform_table(data)
    else:
        return {key: data}


def transform_table(table):
    """
    Unsqueeze keys aligning with values

    Args:
        table: table defining key-value pairs

    Returns:
        - unsqueezed key-value dict
    """
    keys, values = [], []
    for sample in table:
        ks, vs = [], []
        for k, vals in sample.items():
            ks.extend([k for _ in vals])
            vs.extend(vals)
        keys.append(ks)
        values.append(vs)
    return {'key': keys, 'value': values}


def mask_seq(seq: List, p: float, mask='<mask>'):
    """
    Randomly mask tokens in sequence

    Args:
        seq: original sequence
        p: mask probability
        mask: mask token

    Returns:
        - sequence with token mask
    """
    seq = [mask if random.random() < p else s for s in seq]
    return seq


def delete_token(seq: List, p: float):
    """
    Randomly drop tokens

    Args:
        seq: original sequence
        p: drop rate

    Returns:
        - sequence with randomly deleted tokens
    """
    seq = [s for s in seq if random.random() > p]
    return seq


def infill_text(seq: List, lam, mask='<mask>'):
    """
    Mask a segment in the sequence

    Args:
        seq: original sequence
        lam: possion lambda
        mask: mask token

    Returns:
        - a masked sequence
    """
    l = np.random.poisson(lam)
    l = min(l, len(seq))
    start = random.randint(0, len(seq) - l)
    end = start + l
    seq = seq[:start] + [mask] + seq[end:]
    return seq


def permute(seq: List):
    """
    Permute a sequence

    Args:
        seq: sequence to be shuffle

    Returns:
        - shuffled sequence
    """
    random.shuffle(seq)
    return seq


def rotate(seq: List):
    """
    Rotate a sequence

    Args:
        seq: a sequence

    Returns:
        - rotated sequence
    """
    idx = random.randint(0, len(seq) - 1)
    seq = seq[idx:] + seq[:idx]
    return seq


def possible_load_json(sample):
    """
    Callback for json data

    Args:
        sample: data in raw format

    Returns:
        sample (dict): a dict of samples consisting of parallel data of different sources
    """
    try:
        sample = json.loads(sample)
    except:
        pass
    finally:
        return sample


def possible_eval(x):
    """
    Eval a value if possible
    """
    try:
        y = eval(x)
        if type(y).__name__ == 'builtin_function_or_method':
            return x
        return y
    except:
        return x
