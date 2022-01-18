class AbstractRateScheduler:
    """
    AbstractRateScheduler is an auxiliary tools for adjust rate.

    Args:
        rate: initial rate
    """

    def __init__(self, rate: float = 0., *args, **kwargs):
        self._rate: float = rate

    def build(self, *args, **kwargs):
        """
        Build rate scheduler
        """
        pass

    def step_update(self, step, *args, **kwargs):
        """
        Update inner rate with outside states at each step

        Args:
            step: training step
        """
        pass

    def step_reset(self, step, *args, **kwargs):
        """
        Reset inner rate with outside states at each step

        Args:
            step: training step
        """
        pass

    def epoch_update(self, epoch, *args, **kwargs):
        """
        Update inner rate with outside states at each epoch

        Args:
            epoch: training epoch
        """
        pass

    def epoch_reset(self, epoch, *args, **kwargs):
        """
        Update inner rate with outside states at each epoch

        Args:
            epoch: training epoch
        """
        pass

    @property
    def rate(self):
        return self._rate
