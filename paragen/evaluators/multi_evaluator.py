from typing import Dict

from paragen.evaluators import AbstractEvaluator, register_evaluator, create_evaluator


@register_evaluator
class MultiTaskEvaluator(AbstractEvaluator):
    """
    MultiTaskEvaluator for evaluation, which wrapped from Evaluator with different situation.

    Args:
        evaluators (dict): evaluator configurations for building multiple evaluators
    """

    def __init__(self,
                 evaluators: Dict,
                 ):
        super().__init__()
        self._evaluator_configs = evaluators

        self._evaluators = None
        self._task_callback = None

    def build(self, generator, dataloaders, tokenizer, task_callback=None, postprocess=None):
        """
        Build evaluators with given arguments.
        Arguments are dispatched to all the evaluators respectively.

        Args:
            generator (paragen.generators.AbstractGenerator): the inference model to generate hypothesis
            dataloaders (dict[paragen.dataloaders.AbstractDataLoader]): a set of dataloaders to evaluate
            tokenizer (paragen.tokenizers.AbstractTokenizer): a tokenizer
            task_callback: building context in task during for evaluation via a callback function
            postprocess: postprocess pipeline to obtain final hypothesis from predicted results (torch.Tensor)
        """
        self._evaluators = {}
        for name, config in self._evaluator_configs.items():
            self._evaluators[name.upper()] = create_evaluator(config)
            self._evaluators[name.upper()].build(generator=generator,
                                                 dataloaders=dataloaders,
                                                 tokenizer=tokenizer,
                                                 task_callback=task_callback,
                                                 postprocess=postprocess)
        self._task_callback = task_callback

    def eval(self):
        """
        Perform evaluation for each task;
        """
        scores = {}
        for name, evaluator in self._evaluators.items():
            self._task_callback(training=False, infering=True)
            states = evaluator.eval()
            scores.update({'{}.{}'.format(name, key): val for key, val in states.items()})
        return scores
