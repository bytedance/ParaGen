from typing import Dict

from torch.cuda.amp import autocast
import torch

from paragen.trainers import register_trainer
from paragen.trainers.trainer import Trainer
from paragen.utils.ops import auto_map_args
from paragen.utils.profiling import ram


@register_trainer
class AdvTrainer(Trainer):
    def _step(self, samples):
        ram.set_flag("adv_mode")
        total_loss = 0
        ram.write("adv_iter", 0)
        with autocast():
            if isinstance(samples['net_input'], Dict):
                samples['net_input'] = auto_map_args(samples['net_input'], self._model_slots)
            samples = auto_map_args(samples, self._criterion_slots)
            loss, logging_states = self._criterion(*samples)
        loss.backward()
        total_loss += loss.detach()
        
        for adv_iter in range(1, 3):
            ram.write("adv_iter", adv_iter)
            loss, logging_states = self._criterion(*samples)
            loss.backward()
            total_loss += loss.detach()

        total_loss /= 3

        self._optimizer._optimizer.step()
        self._optimizer._optimizer.zero_grad()

        if torch.isnan(loss).any():
            logging_states = {}
        ram.pop("adv_iter")
        ram.reset_flag("adv_mode")
        return logging_states
