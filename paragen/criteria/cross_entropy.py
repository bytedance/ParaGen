import torch
import torch.nn as nn
import torch.nn.functional as F

from paragen.criteria import register_criterion
from paragen.criteria.base_criterion import BaseCriterion


@register_criterion
class CrossEntropy(BaseCriterion):
    """
    Cross Entropy Loss.

    """

    def __init__(self, weight=None, logging_metric='acc'):
        super().__init__()
        self._weight = torch.FloatTensor(weight) if weight is not None else weight
        self._logging_metric = logging_metric
        self._padding_idx = None
        self._nll_loss = None

    def _build(self, model, padding_idx=-1):
        """
        Build a cross entropy loss over model.

        Args:
            model: a neural model for compute cross entropy.
            padding_idx: labels of padding_idx are all ignored to computed nll_loss
        """
        self._model = model
        self._padding_idx = padding_idx
        self._nll_loss = nn.NLLLoss(weight=self._weight, ignore_index=padding_idx)

    def compute_loss(self, lprobs, target):
        """
        Compute loss from a batch of samples

        Args:
            lprobs: neural network output logits
            target: oracle target for a network input

        Returns:
            - loss for network backward and optimization
            - logging information
        """
        lprobs = F.log_softmax(lprobs, dim=-1)

        # compute nll loss
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.view(-1)
        nll_loss = self._nll_loss(lprobs, target)

        # record logging
        logging_states = {
            'loss': nll_loss.data.item(),
        }
        if self._logging_metric == 'acc':
            correct = (lprobs.max(dim=-1)[1] == target).sum().data.item()
            tot = target.size(0)
            logging_states['acc'] = correct / tot
        elif self._logging_metric == 'ppl':
            logging_states['ppl'] = 2 ** (nll_loss.data.item())
        return nll_loss, logging_states
