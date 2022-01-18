from typing import Dict, List, Tuple

from paragen.criteria import AbstractCriterion


class BaseCriterion(AbstractCriterion):
    """
    BaseCriterion is the base class for all the criterion within ParaGen.
    """

    def __init__(self):
        super().__init__()

    def forward(self, net_input, net_output):
        """
        Compute loss from a batch of samples

        Args:
            net_input: neural network input and is used for compute the logits
            net_output (dict): oracle target for a network input
        Returns:
            tuple:
                - **loss**: loss for network backward and optimization
                - **logging_states**: logging information
        """
        if isinstance(net_input, Dict):
            lprobs = self._model(**net_input)
        elif isinstance(net_input, List) or isinstance(net_input, Tuple):
            lprobs = self._model(*net_input)
        else:
            lprobs = self._model(net_input)
        # fetch target with default index 0
        loss, logging_states = self.compute_loss(lprobs, **net_output)
        return loss, logging_states

    def compute_loss(self, *args, **kwargs):
        """
        Compute loss from model results
        """
        raise NotImplementedError

