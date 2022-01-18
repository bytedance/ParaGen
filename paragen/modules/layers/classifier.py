from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from paragen.modules.layers.feed_forward import FFN
from paragen.modules.utils import get_activation_fn
from paragen.utils.runtime import Environment


class LinearClassifier(nn.Module):
    """
    Classifier with only on a linear projection.

    Args:
        d_model: feature dimensionality
        labels: number of classes
        invalid_classes (List): class that is not allowed to produce
    """

    def __init__(self,
                 d_model,
                 labels,
                 invalid_classes: List = None):
        super().__init__()
        self._linear = nn.Linear(d_model, labels, bias=False)
        self._invalid_class_mask = get_invalid_class_mask(labels, invalid_classes) if invalid_classes else None

    def forward(self, x):
        """
        Args:
            x: feature to predict labels
                :math:`(*, D)`, where D is the feature dimension

        Returns:
            - log probability of each classes
                :math: `(*, L)`, where L is the number of classes
        """
        logits = self._linear(x)
        if self._invalid_class_mask is not None:
            logits = logits.masked_fill(self._invalid_class_mask, float('-inf'))
        logits = F.log_softmax(logits, dim=-1)
        return logits


class Classifier(nn.Module):
    """
    Classifier with a feed-forward network projection.

    Args:
        d_model: feature dimensionality
        labels: number of classes
        dim_feedforward: dimensionality of feed forward hidden space
        activation: activation function used in the feed-forward network
        invalid_classes (List): class that is not allowed to produce
    """

    def __init__(self,
                 d_model,
                 labels,
                 dim_feedforward=None,
                 activation="relu",
                 invalid_classes=None,):
        super().__init__()
        self._ffn = FFN(d_model=d_model, dim_feedforward=dim_feedforward, dim_out=labels, activation=activation)

        self._invalid_class_mask = get_invalid_class_mask(labels, invalid_classes) if invalid_classes else None

    def forward(self, x):
        """
        Args:
            x: feature to predict labels
                :math:`(*, D)`, where D is the feature dimension

        Returns:
            - log probability of each classes
                :math: `(*, L)`, where L is the number of classes
        """
        logits = self._ffn(x)
        if self._invalid_class_mask is not None:
            logits = logits.masked_fill(self._invalid_class_mask, float('-inf'))
        logits = F.log_softmax(logits, dim=-1)
        return logits


class HuggingfaceClassifier(nn.Module):
    """
    Classifier implemented in HuggingfaceClassificationHead style.

    Args:
        d_model: feature dimensionality
        labels: number of classes
        inner_dim: dimensionality in the inner vector space.
        activation: activation function used in the feed-forward network
        dropout: dropout rate
    """

    def __init__(self,
                 d_model,
                 labels,
                 inner_dim=None,
                 activation="relu",
                 dropout=0.):
        super().__init__()
        inner_dim = inner_dim or d_model * 2

        self._fc1 = nn.Linear(d_model, inner_dim)
        self._dropout = nn.Dropout(dropout)
        self._fc2 = nn.Linear(inner_dim, labels)
        self._activation = get_activation_fn(activation)

    def forward(self, x):
        """
        Args:
            x: feature to predict labels
                :math:`(*, D)`, where D is the feature dimension

        Returns:
            - log probability of each classes
                :math: `(*, L)`, where L is the number of classes
        """
        x = self._dropout(x)
        x = self._fc1(x)
        x = self._activation(x)
        x = self._dropout(x)
        x = self._fc2(x)
        return x


def get_invalid_class_mask(classes: int, invalid_classes: List):
    """
    Create mask for invalid classes

    Args:
        classes: number of labels
        invalid_classes: invalid class list

    Returns:
        - mask for invalid class
            :math:`(1, L)` where L is the number of classes
    """
    invalid_class_mask = torch.zeros(classes).bool()
    if invalid_classes:
        for idx in invalid_classes:
            invalid_class_mask[idx] = True
    invalid_class_mask = invalid_class_mask.unsqueeze(dim=0)

    env = Environment()
    if env.device.startswith('cuda'):
        invalid_class_mask = invalid_class_mask.cuda()

    return invalid_class_mask

