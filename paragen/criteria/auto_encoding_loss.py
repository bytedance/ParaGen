import torch
import torch.nn.functional as F

from paragen.criteria import register_criterion
from paragen.criteria.base_criterion import BaseCriterion
from paragen.utils.rate_schedulers import create_rate_scheduler


@register_criterion
class AutoEncodingLoss(BaseCriterion):
    """
    Label smoothed cross entropy

    Args:
        epsilon: label smoothing rate
    """

    def __init__(self, epsilon=0.1, beta=1.):
        super().__init__()
        self._epsilon = epsilon
        self._beta_configs = beta

        self._padding_idx = None
        self._beta = None

    def _build(self, model, padding_idx=-1):
        self._padding_idx = padding_idx
        self._model = model

        self._beta = create_rate_scheduler(self._beta_configs)
        self._beta.build()

    def _reconstruct_loss(self, lprobs, target, reduce=False):
        assert target.dim() == lprobs.dim() - 1

        lprobs, target = lprobs.view(-1, lprobs.size(-1)), target.view(-1)
        padding_mask = target.eq(self._padding_idx)
        ntokens = (~padding_mask).sum()
        # calculate nll loss
        nll_loss = -lprobs.gather(dim=-1, index=target.unsqueeze(dim=-1)).squeeze(dim=-1)
        nll_loss.masked_fill_(padding_mask, 0.)
        if reduce:
            nll_loss = nll_loss.sum() / ntokens

        # calculate smoothed loss
        smooth_loss = -lprobs.mean(dim=-1)
        smooth_loss.masked_fill_(padding_mask, 0.)
        smooth_loss = smooth_loss.sum() / ntokens

        return nll_loss, ntokens, smooth_loss

    def step_update(self, step):
        """
        Perform step-level update

        Args:
            step: running step
        """
        self._beta.step_update(step)

    def compute_loss(self, lprobs, net_output):
        """
        Compute loss from a batch of samples

        Args:
            lprobs: neural network output logits
            net_output: neural net output
        Returns:
            - loss for network backward and optimization
            - logging information
        """
        lprobs = F.log_softmax(lprobs, dim=-1)
        # fetch target with default index 0
        target = net_output[0]

        bsz, sql = target.size()
        rec_loss, n_tokens, smooth_loss = self._reconstruct_loss(lprobs, target, reduce=False)
        rec_loss = torch.sum(rec_loss.view(bsz, -1), dim=-1)
        reg_loss = self._model.reg_loss()

        loss = torch.sum(rec_loss + self._beta.rate * reg_loss) / n_tokens
        loss = (1. - self._epsilon) * loss + self._epsilon * smooth_loss if self.training else loss
        
        nll_loss = torch.sum(self._model.nll(rec_loss, reg_loss)) / n_tokens  # real nll loss
        
        logging_states = {
            'reg_weight': self._beta.rate,
            'loss': loss.data.item(),
            'nll_loss': nll_loss.data.item(),
            'ppl': 2 ** (nll_loss.data.item()),
            'reg_loss': torch.mean(reg_loss).item(),
            'rec_loss': (torch.sum(rec_loss)/n_tokens).item()
        }

        return loss, logging_states
