import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class DynamicLinearCombinationLayer(nn.Module):
    """
    DLCL: make the input to be a linear combination of previous outputs
    For pre-norm, x_{l+1} = \sum_{k=0}^{l} W_{k} * LN(y_{k}) 
    For post-norm, x_{l+1} = LN(\sum_{k=0}^{l} W_{k} * y_{k})
    where x_{l}, y_{l} are the input and output of l-th layer

    For pre-norm, LN should be performed in previous layer
    For post-norm, LN is performed in this layer 

    Args:
        idx: this is the `idx`-th layer
        post_ln: post layernorm
    """
    def __init__(self, idx, post_ln=None):
        super(DynamicLinearCombinationLayer, self).__init__()
        assert (idx > 0)
        self.linear = nn.Linear(idx, 1, bias=False)
        nn.init._no_grad_fill_(self.linear.weight, 1.0 / idx)
        self.post_ln = post_ln

    def forward(self, y):
        """
        Args:
            y: SequenceLength x BatchSize x Dim x idx
        """
        x = self.linear(y)
        x = x.squeeze(dim=-1)
        if self.post_ln is not None:
            x = self.post_ln(x)
        return x
