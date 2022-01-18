from paragen.utils.rate_schedulers import AbstractRateScheduler, register_rate_scheduler


@register_rate_scheduler
class ConstantRateScheduler(AbstractRateScheduler):
    """
    ConstantRateScheduler do no schedule rate.

    Args:
        rate: constant rate
    """

    def __init__(self, rate):
        super().__init__(rate)
