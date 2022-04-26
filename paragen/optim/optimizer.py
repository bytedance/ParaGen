from contextlib import contextmanager

import torch

from paragen.utils.runtime import Environment
from paragen.utils.tensor import GradScalerSingleton


class Optimizer:
    """
    Optimizer to optimize a model with its loss

    Args:
        optimizer (torch.optim.optimizer.Optimzer): torch optimizer
        lr_scheduler: learning rate scheduler
        clip_norm: norm for gradient clipping
    """

    def __init__(self, model, optimizer, lr_scheduler, enable_apex=False, update_frequency=1, clip_norm=0.):
        self._enable_apex= enable_apex
        self._update_frequency = update_frequency
        self._clip_norm = clip_norm

        self._model = model
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._env = Environment()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""
        for p in self.params:
            if p.grad is not None:
                if torch.is_tensor(c):
                    c = c.to(p.grad.device)
                p.grad.data.mul_(c)

    def step(self):
        if self._update_frequency > 1:
            self.multiply_grads(1. / self._update_frequency)

        if self._clip_norm > 0:
            grads = [p.grad for p in self._model.parameters() if p.grad is not None]
            total_norm = torch.norm(
                torch.stack(
                    [torch.norm(g, p=2, dtype=torch.float32).to(grads[0]) for g in grads]
                )
            )
            clip_coef = (self._clip_norm / (total_norm + 1e-6)).clamp_(max=1)
            self.multiply_grads(clip_coef)

        if self._env.fp16 and not self._enable_apex:
            with self._possible_skip_synchronize():
                grad_scaler = GradScalerSingleton()
                grad_scaler.step(self._optimizer)
                grad_scaler.update()
        elif self._enable_apex:
            with self._possible_skip_synchronize():
                self._optimizer.step()
        else:
            self._optimizer.step()

    @contextmanager
    def _possible_skip_synchronize(self):
        if self._env.distributed_world > 1 and self._env.distributed in ['horovod', 'hvd']:
            with self._optimizer.skip_synchronize():
                yield
        else:
            yield

    def step_update(self, step, *args, **kwargs):
        """
        Update states at step level

        Args:
            step: training step
        """
        self._lr_scheduler.step_update(step)
        self._update_lr()

    def _update_lr(self):
        """
        Update learning rate for all the parameter groups
        """
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = self._lr_scheduler.rate

    def state_dict(self):
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self._optimizer.load_state_dict(state_dict)

    @property
    def lr(self):
        return self.param_groups[0]['lr']

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def params(self):
        for param_group in self.param_groups:
            for p in param_group["params"]:
                yield p

    @property
    def param_groups(self):
        return self._optimizer.param_groups

    @property
    def update_frequency(self):
        return self._update_frequency
