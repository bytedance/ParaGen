from typing import Tuple

from paragen.criteria import AbstractCriterion, register_criterion


@register_criterion
class SelfContainedLoss(AbstractCriterion):
    """
    SelfContainedLoss.

    """

    def __init__(self):
        super().__init__()

    def build(self, model, *args, **kwargs):
        """
        Build a cross entropy loss over model.

        Args:
            model: a neural model for compute cross entropy.
        """
        self._model = model

    def forward(self, net_input):
        """
        Compute loss via model itself

        Args:
            net_input (dict): neural network input and is used for compute the logits
        Returns:
            - loss for network backward and optimization
            - logging information
        """
        output = self._model.loss(**net_input)
        if isinstance(output, Tuple):
            assert len(output) == 2, 'if a tuple returned, it must be (loss, logging_states)'
            loss, logging_states = output
        else:
            loss = output
            logging_states = {'loss': loss.data.item()}
        return loss, logging_states
