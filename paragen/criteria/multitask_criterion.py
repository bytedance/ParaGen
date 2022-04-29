from typing import Dict

from paragen.criteria import AbstractCriterion, create_criterion, register_criterion


@register_criterion
class MultiTaskCriterion(AbstractCriterion):
    """
    Criterion is the base class for all the criterion within ParaGen.
    """

    def __init__(self, criterions):
        super().__init__()
        self._criterion_configs = criterions

        self._names = [name for name in self._criterion_configs]
        self._criterions, self._weights = None, None

    def _build(self, model, **kwargs):
        """
        Build multi-task criterion by dispatch args to each criterion

        Args:
            model: neural model
            padding_idx: pad idx to ignore
        """
        self._model = model
        self._criterions, self._weights = {}, {}
        for name in self._names:
            criterion_config = self._criterion_configs[name]
            self._weights[name] = criterion_config.pop('weight') if 'weight' in criterion_config else 1
            self._criterions[name] = create_criterion(self._criterion_configs[name])
            self._criterions[name].build(model, **kwargs[name] if name in kwargs else kwargs)

    def forward(self, net_input, net_output):
        """
        Compute loss from a batch of samples

        Args:
            net_input: neural network input and is used for compute the logits
            net_output (dict): oracle target for a network input
        Returns:
            - loss for network backward and optimization
            - logging information
        """
        lprobs_dict = self._model(**net_input)
        assert isinstance(lprobs_dict, Dict), 'A multitask learning model must return a dict of log-probability'
        return self.compute_loss(lprobs_dict, **net_output)

    def compute_loss(self, lprobs_dict, **net_output):
        # fetch target with default index 0
        tot_loss, complete_logging_states = 0, {}
        for name in self._names:
            lprobs, net_out, criterion = lprobs_dict[name], net_output[name], self._criterions[name]
            loss, logging_states = criterion.compute_loss(lprobs, **net_out)
            tot_loss += self._weights[name] * loss
            logging_states = {f'{name}.{key}': val for key, val in logging_states.items()}
            complete_logging_states.update(logging_states)
        complete_logging_states['loss'] = tot_loss.data.item()
        return tot_loss, complete_logging_states

