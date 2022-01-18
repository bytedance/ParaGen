import torch.nn.functional as F

from paragen.criteria import register_criterion
from paragen.criteria.base_criterion import BaseCriterion


@register_criterion
class FocalCrossEntropy(BaseCriterion):
    """
    Label smoothed cross entropy

    Args:
        gamma: focal loss rate
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self._gamma = gamma

        self._padding_idx = None

    def _build(self, model, padding_idx=-1):
        """
        Build a label smoothed cross entropy loss over model.

        Args:
            model: a neural model for compute cross entropy.
            padding_idx: labels of padding_idx are all ignored to computed nll_loss
        """
        self._model = model
        self._padding_idx = padding_idx

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
        correct = (lprobs.max(dim=-1)[1] == target).sum().data.item()
        tot = target.size(0)

        target_padding_mask = target.eq(self._padding_idx)
        assert target.dim() == lprobs.dim() - 1

        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.view(-1)
        target_padding_mask = target_padding_mask.view(-1)
        ntokens = (~target_padding_mask).sum()

        # calculate nll loss
        lprobs = lprobs.gather(dim=-1, index=target.unsqueeze(dim=-1)).squeeze(dim=-1)
        weight = (1 - lprobs.exp()) ** self._gamma
        loss = - weight * lprobs
        loss.masked_fill_(target_padding_mask, 0.)
        loss = loss.sum() / ntokens

        # record logging
        logging_states = {
            'loss': loss.data.item(),
            'ntokens': ntokens.data.item(),
            'acc': correct / tot
        }

        return loss, logging_states
