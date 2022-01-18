from paragen.utils.rate_schedulers import AbstractRateScheduler, register_rate_scheduler


@register_rate_scheduler
class NoamScheduler(AbstractRateScheduler):
    """
    NoamScheduler is a scheduling methods proposed by Noam

    Args:
        d_model: neural model feature dimension
        warmup_steps: training steps in warming up
    """

    def __init__(self, d_model, warmup_steps=4000):
        super().__init__(0.)
        self._warmup_steps = warmup_steps
        self._d_model = d_model

    def step_update(self, step, *args, **kwargs):
        """
        Update inner rate with outside states at each step

        Args:
            step: training step
        """
        self._rate = (self._d_model ** -0.5) * min([step ** -0.5, step * (self._warmup_steps ** -1.5)])
