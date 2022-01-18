class AbstractEvaluator:
    """
    Evaluation scheduler
    """

    def __init__(self, ):
        pass

    def build(self, *args, **kwargs):
        """
        Build evaluator from the given configs and components
        """
        raise NotImplementedError

    def finalize(self):
        """
        Finalize evaluator after finishing evaluation
        """
        raise NotImplementedError

    def _step_reset(self, *args, **kwargs):
        """
        Reset states by step
        """
        pass

    def _step(self, samples):
        """
        Evaluate one batch of samples

        Args:
            samples: a batch of samples
        """
        raise NotImplementedError

    def _step_update(self, *args, **kwargs):
        """
        Update states by step
        """
        pass

    def _eval_reset(self, *args, **kwargs):
        """
        Reset states before the overall evaluation process
        """
        pass

    def _eval_update(self, *args, **kwargs):
        """
        Update states after the overall evaluation process
        """
        pass

    def eval(self):
        """
        Evaluation process
        """
        raise NotImplementedError
