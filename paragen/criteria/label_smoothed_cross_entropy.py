import torch.nn.functional as F

from paragen.criteria import register_criterion
from paragen.criteria.base_criterion import BaseCriterion


@register_criterion
class LabelSmoothedCrossEntropy(BaseCriterion):
    """
    Label smoothed cross entropy

    Args:
        epsilon: label smoothing rate
    """

    def __init__(self, epsilon: float = 0.1):
        super().__init__()
        self._epsilon = epsilon

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
        target_padding_mask = target.eq(self._padding_idx)
        assert target.dim() == lprobs.dim() - 1
        # infer task type
        is_classification_task = len(target.size()) == 1

        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.view(-1)
        target_padding_mask = target_padding_mask.view(-1)
        ntokens = (~target_padding_mask).sum()

        # calculate nll loss
        nll_loss = -lprobs.gather(dim=-1, index=target.unsqueeze(dim=-1)).squeeze(dim=-1)
        nll_loss.masked_fill_(target_padding_mask, 0.)
        nll_loss = nll_loss.sum() / ntokens

        # calculate smoothed loss
        if self._epsilon > 0.:
            smooth_loss = -lprobs.mean(dim=-1)
            smooth_loss.masked_fill_(target_padding_mask, 0.)
            smooth_loss = smooth_loss.sum() / ntokens

            # average nll loss and smoothed loss, weighted by epsilon
            loss = (1. - self._epsilon) * nll_loss + self._epsilon * smooth_loss if self.training else nll_loss
        else:
            loss = nll_loss

        # record logging
        logging_states = {
            'loss': loss.data.item(),
            'nll_loss': nll_loss.data.item(),
            'ntokens': ntokens.data.item(),
        }
        if is_classification_task:
            correct = (lprobs.max(dim=-1)[1] == target).sum().data.item()
            tot = target.size(0)
            logging_states['acc'] = correct / tot
        else:
            logging_states['ppl'] = 2 ** (nll_loss.data.item())

        return loss, logging_states
