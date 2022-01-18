import torch
import torch.nn as nn


class BertLayerNorm(nn.Module):
    """
    BertLayerNorm is layer norm used in BERT.
    It is a layernorm module in the TF style (epsilon inside the square root).

    Args:
        hidden_size: dimensionality of hidden space
    """

    def __init__(self, hidden_size, eps=1e-12):

        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        r"""


        Args:
            x: feature to perform layer norm
                :math:`(*, D)`, where D is the feature dimension

        Returns:
            - normalized feature
                :math:`(*, D)`, where D is the feature dimension
        """
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta
