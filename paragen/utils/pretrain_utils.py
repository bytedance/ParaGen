import random
import torch
import numpy as np
from typing import List


bos_id = 0
pad_id = 1
eos_id = 2
unk_id = 3
mask_id = 3

vocab_size = None


def batch_pad(idx: List[List], pad_len=None) -> List[List]:
    if pad_len is None:
        pad_len = max(map(len, idx))
    return list(map(lambda x: x + [pad_id] * (pad_len - len(x)), idx))

"""
    SBERT   src_tokens:       Thank you <M> <M> me to your party <M> week .
            tgt_tokens:       (original text)

    T5      src_tokens:       Thank you <M1> me to your party <M2> week .
            tgt_tokens:       <M1> for inviting <M2> next

    POS     src_tokens:       Thank you <M1> me to your party <M2> week .
            tgt_tokens:       for inviting next
            tgt_input_tokens: <M1> <M1> <M2>
"""


def apply_pos_mask(src_tokens, mask):
    assert vocab_size is not None
    src_tokens_lst = src_tokens.tolist()
    mask_lst = mask.tolist()

    ret_src, ret_tgt_input, ret_tgt = [], [], []
    for bid, (tokens, tmasks) in enumerate(zip(src_tokens_lst, mask_lst)):
        src, tgt, tgt_input = [], [bos_id], [bos_id]
        sentinel_id = vocab_size
        found = False
        for tid, token in enumerate(tokens):
            if not tmasks[tid]:
                found = False
                src.append(token)
            else:
                if not found:
                    found = True
                    sentinel_id -= 1
                    src.append(sentinel_id)
                tgt_input.append(sentinel_id)
                tgt.append(token)
        tgt_input.append(eos_id)
        tgt.append(eos_id)

        ret_src.append(src)
        ret_tgt_input.append(tgt_input)
        ret_tgt.append(tgt)

    ret_src = torch.tensor(batch_pad(ret_src))
    ret_tgt = torch.tensor(batch_pad(ret_tgt))
    ret_tgt_input = torch.tensor(batch_pad(ret_tgt_input))
    return {
        "src_tokens": ret_src, 
        "tgt_tokens": ret_tgt, 
        "tgt_input_tokens": ret_tgt_input
    }


def apply_t5_mask(src_tokens, mask):
    assert vocab_size != -1
    src_tokens_lst = src_tokens.tolist()
    mask_lst = mask.tolist()

    ret_src, ret_tgt = [], []
    for bid, (tokens, tmasks) in enumerate(zip(src_tokens_lst, mask_lst)):
        src, tgt = [], [bos_id]
        sentinel_id = vocab_size
        found = False
        for tid, token in enumerate(tokens):
            if not tmasks[tid]:
                found = False
                src.append(token)
            else:
                if not found:
                    found = True                
                    sentinel_id -= 1
                    src.append(sentinel_id)
                    tgt.append(sentinel_id)
                tgt.append(token)
        tgt.append(eos_id)
        ret_src.append(src)
        ret_tgt.append(tgt)

    ret_src = torch.tensor(batch_pad(ret_src))
    ret_tgt = torch.tensor(batch_pad(ret_tgt))
    return {
        "src_tokens": ret_src, 
        "tgt_tokens": ret_tgt
    }


def apply_sbert_mask(src_tokens, mask):
    return {
        "src_tokens": src_tokens.masked_fill(mask, mask_id),
        "tgt_tokens": src_tokens.clone()
    }


def generate_span_mask(src_tokens, mask_prob=0.15, possion_lambda=3, cutoff=(1, 10)):
    ori_size = src_tokens.size()
    src_tokens = src_tokens.view(-1, ori_size[-1])
    src_tokens_np = src_tokens.cpu().numpy().copy()

    for bid in range(src_tokens_np.shape[0]):
        seq_len = src_tokens_np.shape[1]
        mask_len = seq_len * mask_prob
        sidxs = iter(np.random.permutation(seq_len))
        for trial in range(3):
            slens = np.random.poisson(possion_lambda, seq_len)
            slens[slens < cutoff[0]] = cutoff[0]
            slens[slens > cutoff[1]] = cutoff[1]
            slens = slens[slens.nonzero()]
            slens = slens[slens.cumsum() < mask_len]
            if len(slens) != 0:
                break
        for slen in slens:
            for trial in range(3):
                sid = next(sidxs)
                lid = sid - 1      # do not merge two spans
                rid = sid + slen   # do not merge two spans
                if lid >= 0 and rid < seq_len and src_tokens_np[bid][lid] != -1 and src_tokens_np[bid][rid] != -1:
                    src_tokens_np[bid][sid: sid + slen] = -1
                    break
    mask = src_tokens_np == -1
    mask = torch.tensor(mask)
    mask = mask.view(ori_size)
    return mask


def apply_word_shuffle(src_tokens, mask):
    assert vocab_size != -1
    src_tokens_lst = src_tokens.tolist()
    mask_lst = mask.tolist()

    ret_src = []
    for bid, (tokens, tmasks) in enumerate(zip(src_tokens_lst, mask_lst)):
        src = []
        local_tokens = []
        for tid, token in enumerate(tokens):
            if not tmasks[tid]:
                if len(local_tokens) != 0:
                    random.shuffle(local_tokens)
                    src.extend(local_tokens)
                    local_tokens = []
                src.append(token)
            else:
                local_tokens.append(token)
        assert len(local_tokens) == 0, "Span MASK should not be in the boundary!"
        ret_src.append(src)

    ret_src = torch.tensor(batch_pad(ret_src))
    return {
        "src_tokens": ret_src, 
        "tgt_tokens": src_tokens.clone()
    }



if __name__ == "__main__":
    from pprint import pprint
    vocab_size = 999
    # tokens = torch.arange(128).repeat(1, 1).long() + 100
    tokens = torch.tensor([[    0,   935, 20751,  7068,  8337,     2,     2]])
    mask = generate_span_mask(tokens)
    print(mask)

    m = apply_sbert_mask(tokens, mask)
    pprint(m)
    print()

    m = apply_t5_mask(tokens, mask)
    pprint(m)
    print()

    m = apply_pos_mask(tokens, mask)
    pprint(m)
    print()

    shuffle_mask = generate_span_mask(tokens, mask_prob=0.1, cutoff=(2, 5))
    span_mask = generate_span_mask(tokens)
    ret0 = apply_word_shuffle(tokens, shuffle_mask)
    ret = apply_sbert_mask(ret0['src_tokens'], span_mask)
    ret["tgt_tokens"] = tokens
    pprint(ret)
    print()
