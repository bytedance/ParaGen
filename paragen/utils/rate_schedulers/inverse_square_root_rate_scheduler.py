from paragen.utils.rate_schedulers import AbstractRateScheduler, register_rate_scheduler


@register_rate_scheduler
class InverseSquareRootRateScheduler(AbstractRateScheduler):
    """
    InverseSquareRootRateScheduler first linearly warm up rate and decay the rate in square root.

    Args:
        rate: maximum rate
        warmup_steps: number of updates in warming up
    """

    def __init__(self, rate, warmup_steps=1000):
        super().__init__(rate)
        self._warmup_steps = warmup_steps

        self._lr_step, self._decay_factor = None, None

    def build(self):
        """
        Build rate scheduler
        """
        self._lr_step = self._rate / self._warmup_steps
        self._decay_factor = self._rate * self._warmup_steps ** 0.5
        self._rate = 0.

    def step_update(self, step, *args, **kwargs):
        """
        Update inner rate with outside states at each step

        Args:
            step: training step
        """
        if step < self._warmup_steps:
            self._rate = step * self._lr_step
        else:
            self._rate = self._decay_factor * step ** -0.5

