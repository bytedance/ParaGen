from collections import OrderedDict
from contextlib import contextmanager
from typing import Dict
import inspect


def echo(x):
    return x


def merge_states(exist, current, weight=None):
    """
    Merge a new dict into a historical one

    Args:
        exist: long-time dict
        current: dict info at current time
        weight: weight on current dict

    Returns:
        - a merge long-time dict
    """
    if not current:
        return exist
    for name, val in current.items():
        if name not in exist:
            exist[name] = val
        else:
            if weight is not None:
                exist[name] = exist[name] * (1 - weight) + val * weight
            else:
                exist[name] += val
    return exist


def recursive(fn):
    """
    Make a function to work recursively, regardless dict, list and tuple

    Args:
        fn: processing function

    Returns:
        - a recursive version of given function
    """

    def rfn(x, *args, **kwargs):
        if isinstance(x, dict):
            return {key: rfn(val, *args, **kwargs) for key, val in x.items()}
        elif isinstance(x, list):
            return [rfn(val, *args, **kwargs) for val in x]
        elif isinstance(x, tuple):
            return tuple([rfn(val, *args, **kwargs) for val in x])
        else:
            return fn(x, *args, **kwargs)

    return rfn


def get_ordered_values_from_table_by_key(table, reverse=False):
    """
    Get value list where the value orders are determined by their keys.

    Args:
        table: a table of data
        reverse: value list in a reversed order

    Returns:
        - an ordered list of values
    """
    keys = [_ for _ in table]
    keys.sort(reverse=reverse)
    values = [table[k] for k in keys]
    return values


def auto_map_args(d: Dict, slots: OrderedDict):
    """
    Auto map a dict of data to a pre-defined slots

    Args:
        d: a dict of data
        slots: pre-defined slots

    Returns:
        - a tuple of data, where the order of data corresponds to the key orders in slots
    """
    kwargs = OrderedDict()
    for key, val in slots.items():
        kwargs[key] = val
    for key, val in d.items():
        kwargs[key] = val
    args = tuple([v for _, v in kwargs.items()])
    while len(args) > 0:
        if args[-1] is None:
            args = args[:-1]
        else:
            break
    return args


def inspect_fn(fn):
    """
    Inspect arguments of a function

    Args:
        fn: a function to inspect

    Returns:
        - an ordered dict with arguments and defaulted values
    """
    args = OrderedDict()
    signature = inspect.signature(fn)
    for key, val in signature.parameters.items():
        if key not in ['args', 'kwargs']:
            args[key] = val.default
    return args


def auto_map(kwargs, fn):
    """
    Auto map function input to function arguments

    Args:
        kwargs: function input
        fn: a function

    Returns:
        - a tuple of function inputs
    """
    return auto_map_args(kwargs, inspect_fn(fn))


@contextmanager
def local_seed(seed):
    """
    Set local running context with a given seed, and recover the seed once exited.

    Args:
        seed: seed in local context
    """
    import torch
    from paragen.utils.runtime import Environment
    state = torch.random.get_rng_state()
    env = Environment()
    if env.device == 'cuda':
        state_cuda = torch.cuda.random.get_rng_state()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(state)
        if env.device == 'cuda':
            torch.cuda.random.set_rng_state(state_cuda)


deepcopy_on_ref = recursive(lambda x: x)


def search_key(d, key):
    if key in d:
        return d[key]
    else:
        for k, v in d.items():
            if isinstance(v, Dict):
                return search_key(v, key)
    return None
