from contextlib import contextmanager


class AbstractTrainer:
    """
    Training scheduler

    Args:
        max_epochs (int): training epoch
        max_steps (int): maximum traing steps
        validate_interval_epoch (int): epoch-level validation interval
        validate_interval_step (int): step-level validation interval
        start_validate_epoch (int): epoch when starting validation
        start_validate_step (int): step when starting validation
    """

    def __init__(self,
                 max_epochs=0,
                 max_steps=0,
                 validate_interval_epoch=None,
                 validate_interval_step=None,
                 save_steps=None,
                 save_epochs=None,
                 start_validate_epoch=0,
                 start_validate_step=0,
                 log_interval=500):
        self._max_epochs = max_epochs
        self._max_steps = max_steps
        self._validate_interval_epoch = validate_interval_epoch
        self._validate_interval_step = validate_interval_step
        self._save_steps = save_steps or validate_interval_step
        self._save_epochs = save_epochs or validate_interval_epoch
        self._start_validate_epoch = start_validate_epoch
        self._start_validate_step = start_validate_step
        self._log_interval = log_interval

    def build(self, *args, **kwargs):
        """
        Build trainer from the given configs and components
        """
        raise NotImplementedError

    def train(self, *args, **kwargs):
        """
        Training process
        """
        raise NotImplementedError

    def _epoch_train(self, *args, **kwargs):
        """
        Train model in a epoch
        """
        raise NotImplementedError

    def _safe_step(self, samples):
        """
        Safely step a batch of samples

        Args:
            samples: a set of batches
        """
        raise NotImplementedError

    def _step(self, samples):
        """
        Train a set of batches with only one gradient update.

        Args:
            samples: a set of batches
        """
        raise NotImplementedError

    @contextmanager
    def _epoch_context(self, *args, **kwargs):
        """
        Defines context processing before and after training on each epoch
        """
        yield

    @contextmanager
    def _step_context(self, *args, **kwargs):
        """
        Defines context processing before and after training on each step
        """
        yield

    def _update_logging(self):
        """
        update global states with states at one step
        """
        pass

    def _eval(self):
        """
        Evaluate the model
        """
        pass

    def _eval_by_criterion(self):
        """
        Evaluate model with training criterion

        Returns:
            loss: average development loss on the given dataloder
        """
        pass

    def _eval_dataset_by_criterion(self, dataloader):
        """
        Evaluate model with training criterion

        Args:
            dataloader: a evaluation dataloader

        Returns:
            loss: average development loss on the given dataloder
        """
        pass

    def _eval_by_evaluator(self):
        """
        Evaluate model in inference

        Returns:
            scores: score produced by evalutor
        """
        pass

    def _save_model(self, **kwargs):
        """
        save model

        Args:
            **kwargs: saving information for compute checkpoint name
        """
        pass

    def set_mode(self, mode):
        """
        Switch mode ['train', 'valid', 'infer', ...] of a trainer

        Args:
            mode: mode
        """
        pass
