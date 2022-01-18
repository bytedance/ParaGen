import torch.nn as nn

from paragen.criteria import register_criterion
from paragen.criteria.base_criterion import BaseCriterion


@register_criterion
class MSE(BaseCriterion):
    """
    Mean square error

    """

    def __init__(self):
        super().__init__()
        self._mse_loss = None

    def _build(self, model):
        """
        Build a cross entropy loss over model.

        Args:
            model: a neural model for compute cross entropy.
        """
        self._model = model
        self._mse_loss = nn.MSELoss()

    def compute_loss(self, pred, target):
        """
        Compute loss from a batch of samples

        Args:
            pred: neural network output
            target: oracle target for a network input
        Returns:
            - loss for network backward and optimization
            - logging information
        """
        # compute nll loss
        pred = pred.view(-1)
        target = target.view(-1)
        mse_loss = self._mse_loss(pred, target)

        # record logging
        logging_states = {
            'loss': mse_loss.data.item(),
        }
        return mse_loss, logging_states
