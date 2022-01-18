import torch.nn as nn


class Embedding(nn.Embedding):
    """
    Embedding is a wrapped class of torch.nn.Embedding with normal initialization on weight
    and zero initialization on pad.

    Args:
        vocab_size: vocabulary size
        d_model: feature dimensionality
        padding_idx: index of pad, which is a special token to ignore
    """

    def __init__(self, vocab_size, d_model, padding_idx=None):
        super().__init__(vocab_size, d_model, padding_idx=padding_idx)
        nn.init.normal_(self.weight, mean=0, std=d_model ** -0.5)
        if padding_idx:
            nn.init.constant_(self.weight[padding_idx], 0)
