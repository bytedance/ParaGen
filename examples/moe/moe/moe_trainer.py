from paragen.trainers.trainer import Trainer
from paragen.trainers import register_trainer


@register_trainer
class MoETrainer(Trainer):
    """
        Args:
            load_balance_alpha: importance of load balacing loss
    """
    def __init__(self,
                 load_balance_alpha=0.,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_balance_alpha = load_balance_alpha

    def _forward_loss(self, samples):
        loss, logging_states = self._criterion(**samples)
        loss, logging_states = self._load_balance_loss(loss, logging_states)
        return loss, logging_states

    def _load_balance_loss(self, loss, logging_states):
        moe_loss = self._model._encoder.moe_loss + self._model._decoder.moe_loss
        moe_loss /= 2
        loss += moe_loss*self._load_balance_alpha
        logging_states['moe_loss'] = moe_loss.data.item()
        return loss, logging_states

