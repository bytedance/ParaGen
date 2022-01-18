import torch.nn as nn

from paragen.modules.utils import get_activation_fn


class FFN(nn.Module):
    """
    Feed-forward neural network

    Args:
        d_model: input feature dimension
        dim_feedforward: dimensionality of inner vector space
        dim_out: output feature dimensionality
        activation: activation function
        bias: requires bias in output linear function
    """

    def __init__(self,
                 d_model,
                 dim_feedforward=None,
                 dim_out=None,
                 activation="relu",
                 bias=True):
        super().__init__()
        dim_feedforward = dim_feedforward or d_model
        dim_out = dim_out or d_model

        self._fc1 = nn.Linear(d_model, dim_feedforward)
        self._fc2 = nn.Linear(dim_feedforward, dim_out, bias=bias)
        self._activation = get_activation_fn(activation)

    def forward(self, x):
        """
        Args:
            x: feature to perform feed-forward net
                :math:`(*, D)`, where D is feature dimension

        Returns:
            - feed forward output
                :math:`(*, D)`, where D is feature dimension
        """
        x = self._fc1(x)
        x = self._activation(x)
        x = self._fc2(x)
        return x
