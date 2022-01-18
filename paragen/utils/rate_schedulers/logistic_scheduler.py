import math

from paragen.utils.rate_schedulers import AbstractRateScheduler, register_rate_scheduler


@register_rate_scheduler
class LogisticScheduler(AbstractRateScheduler):
    """
    LogisticScheduler scheduler the rate with logistic decay.

    Args:
        k: decaying weight
        x0: bias
    """

    def __init__(self, k=0.0025, x0=4000):
        super().__init__(0.)
        self._k = k
        self._x0 = x0

    def step_update(self, step, *args, **kwargs):
        """
        Update inner rate with outside states at each step

        Args:
            step: training step
        """
        self._rate = 1 / (1 + math.exp(-self._k * (step - self._x0)))
