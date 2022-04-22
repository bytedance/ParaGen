import os.path
from collections import OrderedDict
from contextlib import contextmanager
from typing import Dict, List, Tuple
import subprocess
import time
import numpy as np
import logging
logger = logging.getLogger(__name__)

from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.multiprocessing import Process
import torch

from paragen.utils.io import TEMP_IO_SAVE_PATH, wait_until_exist
from paragen.utils.ops import recursive
from paragen.utils.runtime import Environment, singleton


def list2tensor(x):
    if isinstance(x, Dict):
        return {k: list2tensor(v) for k, v in x.items()}
    elif isinstance(x, List):
        _x = get_example_obj(x)
        return create_tensor(x, type(_x))
    else:
        return x


def convert_idx_to_tensor(idx, pad, ndim=None):
    """
    Convert a nd list of indices to a torch tensor

    Args:
        idx: a nd list of indices
        pad: padding index
        ndim: dimension for idx

    Returns:
        - indices in torch tensor
    """
    max_lengths = maxlen(idx, ndim=ndim)
    tensor_type = type(pad)
    ndim = len(max_lengths)
    idx = pad_idx(idx, max_lengths, pad, ndim=ndim)
    idx = create_tensor(idx, tensor_type)
    return idx


def maxlen(idx, ndim=None):
    """
    Compute maxlen tuple from index

    Args:
        idx: a nd list of indices
        ndim: ndim for idx

    Returns:
        - tensor shape (tuple) of index list
    """
    def _max_tuple(tuples: List[Tuple]):
        return tuple(max(sizes) for sizes in zip(*tuples))

    if ndim is None:
        if isinstance(idx, list):
            tuples = [maxlen(i) for i in idx]
            return (len(idx),) + _max_tuple(tuples)
        else:
            return tuple()
    else:
        if ndim > 1:
            tuples = [maxlen(i, ndim-1) for i in idx]
            return (len(idx),) + _max_tuple(tuples)
        else:
            return len(idx),


def pad_idx(idx, max_lengths, pad_id, ndim):
    """
    Complete index list to a certain shape with padding

    Args:
        idx: a nd list of indices
        max_lengths: n-size tuple defining shape
        pad_id: padding index
        ndim: dimension for idx

    Returns:
        - a nd list of indices with padding
    """
    if ndim > 1:
        l, suff = max_lengths[0], max_lengths[1:]
        content = [pad_idx(i, suff, pad_id, ndim-1) for i in idx]
        if len(idx) < l:
            pad = create_pad((l - len(idx),) + suff, pad_id)
            content += pad
        return content
    else:
        return idx + [pad_id for _ in range(max_lengths[0] - len(idx))]


def create_pad(size, pad_id):
    """
    Create a padding list of a given size

    Args:
        size: nd list shape
        pad_id: padding index

    Returns:
        - padding list of the given size
    """
    if len(size) == 1:
        return [pad_id for _ in range(size[0])]
    else:
        return [create_pad(size[1:], pad_id) for _ in range(size[0])]


def create_tensor(idx: List, tensor_type) -> Tensor:
    """
    Create torch tensor from index

    Args:
        idx: index list
        tensor_type: type of tensor

    Returns:
        - a torch tensor created from index
    """
    if tensor_type is int:
        T = torch.LongTensor(idx)
    elif tensor_type is float:
        T = torch.FloatTensor(idx)
    elif tensor_type is bool:
        T = torch.BoolTensor(idx)
    else:
        raise TypeError
    return T


def convert_tensor_to_idx(tensor: Tensor, bos: int = None, eos: int = None, pad: int = None):
    """
    Convert a tensor to index.

    Args:
        tensor: original tensor
        bos: begin-of-sequence index
        eos: end-of-sequence index
        pad: padding index

    Returns:
        - a nd list of indices
    """
    idx = tensor.tolist()
    if bos is not None and eos is not None and pad is not None:
        idx = remove_special_tokens(idx, bos, eos, pad)
    return idx


def remove_special_tokens(idx, bos: int, eos: int, pad: int):
    """
    Remove special tokens from nd index list

    Args:
        idx: a nd index list
        bos: begin-of-sequence index
        eos: end-of-sequence index
        pad: padding index

    Returns:
        - index list without special tokens
    """
    if isinstance(idx, list) and isinstance(idx[0], int):
        return [i for i in idx if i not in [bos, eos, pad]]
    else:
        return [remove_special_tokens(i, bos, eos, pad) for i in idx]


def find_eos(idx: list, eos: int):
    """
    Find eos position

    Args:
        idx: index list
        eos: end-of-sequence index

    Returns:
        - position of eos
    """
    for pos, i in enumerate(idx):
        if i == eos:
            return pos
    return None


def _to_device(tensor, device, fp16=False):
    """
    Move a tensor to device

    Args:
        tensor: original tensor
        device: device name
        fp16: whether to perform fp16

    Returns:
        - tensor on the given device
    """
    if isinstance(tensor, torch.Tensor):
        if device.startswith('cuda'):
            tensor = tensor.cuda()
            if isinstance(tensor, torch.FloatTensor) and fp16:
                tensor = tensor.half()
        elif device == 'cpu':
            tensor = tensor.cpu()
    return tensor


def half_samples(samples):
    """
    Half tensor of the given samples

    Args:
        samples: samples to half

    Returns:
        - halved samples
    """
    if isinstance(samples, List):
        halved = []
        is_dummy = False
        for s in samples:
            hs, dummy = half_samples(s)
            is_dummy = dummy or is_dummy
            halved.append(hs)
        return halved, is_dummy
    elif isinstance(samples, Dict):
        t = get_example_obj(samples)
        size = t.size(0)
        idx = np.random.choice(list(range(size)), size // 2, replace=False)
        if len(idx) > 0:
            index = recursive(index_tensor)
            return index(samples, idx), False
        else:
            dummy = recursive(dummy_tensor)
            return dummy(samples), True
    else:
        raise NotImplementedError


def split_samples(samples):
    """
    Half tensor of the given samples

    Args:
        samples: samples to half

    Returns:
        - two halved samples
    """
    if isinstance(samples, List):
        samples = [split_samples(s) for s in samples]
        samples = list(zip(*samples))
        return samples[0], samples[1]
    elif isinstance(samples, Dict):
        samples = {k: split_samples(v) for k, v in samples.items()}
        return {k: v[0] for k, v in samples.items()}, {k: v[1] for k, v in samples.items()}
    elif isinstance(samples, Tensor):
        idx = samples.shape[0] // 2
        return samples[:idx], samples[idx:]
    else:
        raise NotImplementedError


def index_tensor(tensor, idx):
    """
    select tensor with the row of given indices

    Args:
        tensor: original
        idx: index to keep

    Returns:
        - tensor with selected row
    """
    return tensor[idx]


def dummy_tensor(tensor):
    size = tensor.size()
    new_size = tuple([1 for _ in size[1:]])
    tot = 1
    for s in size:
        tot *= s
    tensor = tensor.view((tot, ) + new_size)
    tensor = tensor[:1]
    return tensor


def get_example_obj(x):
    """
    Get a example object from List, Tuple or Dict

    Args:
        x: given object

    Returns:
        - an example object
    """
    if isinstance(x, List) or isinstance(x, Tuple):
        return get_example_obj(x[0])
    elif isinstance(x, Dict):
        for v in x.values():
            return get_example_obj(v)
    else:
        return x


@contextmanager
def possible_autocast():
    """
    Possibly perform autocast
    """
    env = Environment()
    if env.fp16:
        with autocast():
            yield
    else:
        yield


@singleton
class GradScalerSingleton:
    """
    GradScaler for fp16 training
    """

    def __init__(self) -> None:
        self._grad_scaler = GradScaler()

    def scale_loss(self, loss):
        return self._grad_scaler.scale(loss)

    def step(self, optimizer):
        self._grad_scaler.step(optimizer)

    def update(self):
        self._grad_scaler.update()


def possible_scale_loss(loss):
    """
    Possibly scale loss in fp training
    """
    env = Environment()
    if env.fp16:
        grad_scaler = GradScalerSingleton()
        return grad_scaler.scale_loss(loss)
    else:
        return loss


def save_avg_ckpt(last_ckpts, save_path, timeout=10000, wait=False):

    def _save(ckpts, path, timeout=10000):
        for ckpt in ckpts:
            if not wait_until_exist(ckpt, timeout=timeout):
                logger.info(f'timeout: {ckpt} not found')
                return
        time.sleep(10)
        avg_state_dict = get_avg_ckpt(ckpts)
        save_ckpt(avg_state_dict, path, wait=True)

    if wait:
        _save(last_ckpts, save_path, timeout)
    else:
        Process(target=_save, args=(last_ckpts, save_path, timeout)).start()


def save_ckpt(state_dict, path, retry=5, wait=False):

    def _save(state_dict, path):
        for _ in range(retry):
            try:
                tmp_path = os.path.join(TEMP_IO_SAVE_PATH, f"tmp.put.{path.split('/')[-1]}")
                with open(tmp_path, 'wb') as fout:
                    torch.save(state_dict, fout)
                if path.startswith('hdfs:'):
                    subprocess.run(["hadoop", "fs", "-put", "-f", tmp_path, path],
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
                    subprocess.run(['rm', tmp_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    subprocess.run(["mv", tmp_path, path],
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
                logger.info(f'successfully save state_dict to {path}')
                break
            except Exception as e:
                logger.warning(f'saving checkpoint {path} fails: {e}')

    state_dict = to_device(state_dict, 'cpu')
    if wait:
        _save(state_dict, path)
    else:
        Process(target=_save, args=(state_dict, path)).start()


def get_avg_ckpt(ckpt_paths, device='cpu'):
    state_dict_list = []
    for path in ckpt_paths:
        if path.startswith('hdfs:'):
            local_path = os.path.join(TEMP_IO_SAVE_PATH, f'tmp.get.{path.split("/")[-1]}')
            subprocess.run(['hadoop', 'fs', '-get', path, local_path])
            with open(local_path, 'rb') as fin:
                state_dict_list.append(torch.load(fin, map_location='cpu')['model'])
            subprocess.run(['rm', local_path])
        else:
            with open(path, 'rb') as fin:
                state_dict_list.append(torch.load(fin, map_location='cpu')['model'])
    state_dict = average_checkpoints(state_dict_list)
    if device != 'cpu':
        state_dict = {k: v.to(device) for k, v in state_dict.items()}
    return {"model": state_dict}


def average_checkpoints(state_dict_list: List):
    state_dict = OrderedDict()
    for i, sd in enumerate(state_dict_list):
        for key in sd:
            p = sd[key]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if i == 0:
                state_dict[key] = p.numpy()
            else:
                state_dict[key] = state_dict[key] + p.numpy()
    ckpt_num = len(state_dict_list)
    for key in state_dict:
        state_dict[key] = state_dict[key] / ckpt_num
        state_dict[key] = torch.from_numpy(state_dict[key])
    return state_dict


to_device = recursive(_to_device)
