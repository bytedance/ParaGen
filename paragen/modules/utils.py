from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation_fn(activation):
    """
    Get activation function by name

    Args:
        activation: activation function name

    Returns:
        - activation function
    """
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise KeyError


def fuse_key_value(key, value, key_padding_mask, value_padding_mask, fusing):
    """
    Fuse key representation and value representation

    Args:
        key:
            :math:`(K, N, E)` where N is the batch size, K is the key number, E is the embedding size.
        value:
            :math:`(L, K, N, E)` where L is the value length, N is the batch size, K is the key number,
            E is the embedding size.
        key_padding_mask:
            :math:`(N, K)` where N is the batch size, K is the key number, E is the embedding size.`
        value_padding_mask:
            :math:`(N, K, L)` where N is the batch size, K is the key number, L is the value length size.`
        fusing: fusing type

    Returns:
        - output: fused representation for key-value pair
    """
    if fusing == 'max-pool-value':
        value, _ = value.max(dim=0)
        return key + value, key_padding_mask
    elif fusing == 'expand-key':
        key = key.unsqueeze(0)
        return key + value, value_padding_mask
    else:
        raise NotImplementedError


def create_init_scores(prev_tokens, tensor):
    """
    Create init scores in search algorithms

    Args:
        prev_tokens: previous token
        tensor: a type tensor

    Returns:
        - initial scores as zeros
    """
    batch_size = prev_tokens.size(0)
    prev_scores = torch.zeros(batch_size).type_as(tensor)
    return prev_scores


def create_upper_triangular_mask(tensor: Tensor):
    """
    Create upper triangular mask. It is usually used in auto-regressive model in training

    Args:
        tensor:
            :math: (N, T, *) where T is target dimension

    Returns:
        - upper triangular mask:
            :math:`(N, T)` where T is target dimension
    """
    sz = tensor.size(1)
    mask = (torch.triu(torch.ones(sz, sz)) == 1).type_as(tensor).bool()
    return mask.detach()


def create_max_segment_mask(tensor: Tensor, max_segment_length):
    """
    Create max-segment mask.

    Args:
        tensor:
            :math: (N, T, *) where T is target dimension

    Returns:
        - max-segment mask:
            :math:`(N, T)` where T is target dimension
    """
    sz = tensor.size(1)
    mask = [[i <= j < i + max_segment_length for j in range(sz)] for i in range(sz)]
    mask = torch.BoolTensor(mask).type_as(tensor).bool()
    return mask


def create_time_mask(tensor: Tensor):
    """
    Create time mask. It is usually used in auto-regressive model in training.

    Args:
        tensor:
            :math: (N, T, *) where T is target dimension

    Returns:
        - upper triangular mask:
            :math:`(N, T)` where T is target dimension
    """
    sz = tensor.size(1)
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1).type_as(tensor).float()
    mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask.detach()


def sample_from_gaussian(mean, logvar):
    """
    Sample a vector from gaussian distribution

    Args:
        mean: mean of gaussian distribution
        logvar: log-variance of gaussian distribution

    Returns:
        - sampled vector from gassian distribution
    """
    std = torch.exp(0.5 * logvar)
    z = torch.randn_like(mean)
    z = z * std + mean
    return z


def mean_pooling(x, x_padding_mask):
    """
    Mean pooling on representation

    Args:
        x: feature matrix
            :math:`(T, N, E)', where T is sequence length, N is batch size and E is feature dimension.
        x_padding_mask:
            :math:`(N, T)`, where T is sequence length and N is batch size.

    Returns:
    """
    sql = torch.sum((~x_padding_mask).long(), -1).unsqueeze(-1) # [bsz, 1]
    return torch.sum(x * (~x_padding_mask).transpose(0, 1).unsqueeze(-1).float(), dim=0) / sql


def create_source_target_modality(d_model,
                                  src_vocab_size,
                                  tgt_vocab_size,
                                  src_padding_idx,
                                  tgt_padding_idx,
                                  share_embedding=None):
    """
    Create source and target modality (embedding)

    Args:
        d_model: model dimension
        src_vocab_size: vocabulary size at source side
        tgt_vocab_size: vocabulary size at target side
        src_padding_idx: padding_idx in source vocabulary
        tgt_padding_idx: padding_idx in target vocabulary
        share_embedding: how the embedding is share [all, decoder-input-output, None].
            `all` indicates that source embedding, target embedding and target
             output projection are the same.
            `decoder-input-output` indicates that only target embedding and target
             output projection are the same.
            `None` indicates that none of them are the same.

    Returns:
        - source embedding
            :math:`(V_s, E)` where V_s is source vocabulary size and E is feature dimension.
        - target embedding
            :math:`(V_t, E)` where V_s is target vocabulary size and E is feature dimension.
        - target output projection
            :math:`(V_t, E)` where V_s is target vocabulary size and E is feature dimension.
    """

    from paragen.modules.layers.embedding import Embedding

    src_embed = Embedding(vocab_size=src_vocab_size,
                          d_model=d_model,
                          padding_idx=src_padding_idx)
    if share_embedding == 'all':
        assert src_vocab_size == tgt_vocab_size, \
            'The sizes of source and target vocabulary must be equal when sharing all the embedding'
        assert src_padding_idx == tgt_padding_idx, \
            'The padding idx must be the same by sharing all the embedding'
        tgt_embed = src_embed
    else:
        tgt_embed = Embedding(vocab_size=tgt_vocab_size,
                              d_model=d_model,
                              padding_idx=tgt_padding_idx)
    if share_embedding in ['all', 'decoder-input-output']:
        tgt_out_proj = nn.Linear(tgt_embed.weight.shape[1],
                                 tgt_embed.weight.shape[0],
                                 bias=False)
        tgt_out_proj.weight = tgt_embed.weight
    else:
        tgt_out_proj = nn.Linear(d_model, tgt_vocab_size, bias=False)
        nn.init.normal_(tgt_out_proj.weight, mean=0, std=d_model ** -0.5)

    return src_embed, tgt_embed, tgt_out_proj


def create_padding_mask_from_length(length, maxlen=None):
    """
    Transform a sequence length matrix to padding mask

    Args:
        length: sequence length matrix
            :math:`(N)` where N is batch size

    Returns:
        - padding mask indicating length
            :math:`(N, L)` where N is batch size and L is maximum length in `length`
    """
    bsz = length.size(0)
    if maxlen is None:
        maxlen = length.max()
    index = torch.arange(maxlen).long().unsqueeze(0).repeat(bsz, 1).to(length)
    padding_mask = index.ge(length.unsqueeze(1))
    return padding_mask


def uniform_assignment(src_padding_mask, tgt_padding_mask):
    """
    Compute uniform assignment matrix between source sequence and target sequence

    Args:
        src_padding_mask: padding mask at source side
            :math:`(N, S)` where N is batch size and S is source sequence length
        tgt_padding_mask: padding mask at target side
            :math:`(N, T)` where N is batch size and T is source sequence length

    Returns:
        - uniform assignment matrix:
            :math:`(N, T, S)` where N is batch size, T is source sequence length and S is source sequence length
    """
    src_length = (~src_padding_mask.bool()).sum(dim=-1)
    tgt_length = (~tgt_padding_mask.bool()).sum(dim=-1)
    max_trg_len = tgt_padding_mask.size(-1)
    steps = (src_length.float() - 1) / (tgt_length.float() - 1 + 1e-4)
    # max_trg_len
    index_t = new_arange(tgt_length, max_trg_len).float()
    index_t = steps[:, None] * index_t[None, :]  # batch_size X max_trg_len
    index_t = torch.round(index_t).long()
    index_t = index_t.masked_fill(tgt_padding_mask, 0)
    return index_t.detach()


def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


def create_sequence(padding_mask, idx, pad_id=None):
    """
    Create a sequence filled with an index

    Args:
        padding_mask: padding mask of target sequence
        idx: filled value
        pad_id: index of pad

    Returns:
        - a long tensor that is of the same shape as padding_mask and filled with idx
    """
    seq = padding_mask.long()
    seq = seq.masked_fill(~padding_mask, idx)
    if pad_id is not None:
        seq = seq.masked_fill(padding_mask, pad_id)
    return seq


def param_summary(model):
    """
    Compute the number of trainable/total parameters

    Args:
        model: a torch module

    Returns:
        - a tuple of number of (trainable, total) parameters
    """
    numel_train = sum(p.numel() for p in model.parameters() if p.requires_grad) // 1000000
    numel_total = sum(p.numel() for p in model.parameters()) // 1000000
    return numel_train, numel_total
