import torch

from paragen.trainers.trainer import Trainer
from paragen.trainers import register_trainer


@register_trainer
class CTCTrainer(Trainer):
    """
    Trainer with ctc strategy
    """

    def _forward_loss(self, samples):
        """
        Train one batch of samples

        Args:
            samples: a batch of samples
        Returns:
            logging_states: states to display in progress bar
        """
        self._model.reset(mode='train')
        # may be we remove set_seed
        self._model.set_seed(self._tot_step_cnt)
        samples['net_input']['target'] = None
        samples['net_input']['fusing_target_mask'] = None

        loss, logging_states = self._criterion(**samples)

        if torch.isnan(loss).any():
            logging_states = {}
        return loss, logging_states
