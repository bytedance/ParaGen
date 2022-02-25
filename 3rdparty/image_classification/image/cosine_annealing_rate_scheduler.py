import math

from paragen.utils.rate_schedulers import AbstractRateScheduler, register_rate_scheduler


@register_rate_scheduler
class CosineAnnealingRateScheduler(AbstractRateScheduler):
    """
    PolynomialDecaySchedulaer first linearly warm up rate, then decay the rate polynomailly and
    finally keep at an minimum rate.

    Args:
        max_rate: maximum rate
        total_steps: total training steps
    """

    def __init__(self,
                 max_rate,
                 warmup_steps=0.,
                 min_rate=0.,
                 total_steps=-1.):
        super().__init__(0.)
        self._max_rate = max_rate
        self._min_rate = min_rate
        self._warmup_steps = warmup_steps
        self._total_steps = total_steps - warmup_steps

        self._k = max_rate - min_rate

    def step_update(self, step, *args, **kwargs):
        """
        Update inner rate with outside states at each step

        Args:
            step: training step
        """
        if step < self._warmup_steps:
            self._rate = (self._max_rate - self._min_rate) / self._warmup_steps * step
        else:
            step -= self._warmup_steps
            self._rate = self._min_rate + 0.5 * (self._max_rate - self._min_rate) * (1 + math.cos(step / self._total_steps * math.pi))
