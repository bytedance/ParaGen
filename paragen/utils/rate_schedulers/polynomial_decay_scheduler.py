from paragen.utils.rate_schedulers import AbstractRateScheduler, register_rate_scheduler


@register_rate_scheduler
class PolynomialDecayScheduler(AbstractRateScheduler):
    """
    PolynomialDecaySchedulaer first linearly warm up rate, then decay the rate polynomailly and
    finally keep at an minimum rate.

    Args:
        max_rate: maximum rate
        total_steps: total training steps
        warmup_steps: number of updates in warming up
        end_rate: minimum rate at end
        power: polynomial decaying power
    """

    def __init__(self, max_rate, total_steps, warmup_steps=4000, end_rate=0.0, power=1.0,):
        super().__init__(0.)
        self._max_rate = max_rate
        self._total_steps = total_steps
        self._warmup_steps = warmup_steps
        self._end_rate = end_rate
        self._power = power

    def step_update(self, step, *args, **kwargs):
        """
        Update inner rate with outside states at each step

        Args:
            step: training step
        """
        if self._warmup_steps > 0 and step <= self._warmup_steps:
            warmup_factor = step / float(self._warmup_steps)
            self._rate = warmup_factor * self._max_rate
        elif step >= self._total_steps:
            self._rate = self._end_rate
        else:
            rate_range = self._max_rate - self._end_rate
            pct_remaining = 1 - (step - self._warmup_steps) / (self._total_steps - self._warmup_steps)
            self._rate = rate_range * pct_remaining ** self._power + self._end_rate
