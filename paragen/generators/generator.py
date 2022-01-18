from paragen.generators import AbstractGenerator, register_generator
from paragen.utils.ops import inspect_fn


@register_generator
class Generator(AbstractGenerator):
    """
    Generator wrap a model with inference algorithms.
    Generator has the same function and interface as model.
    It can be directly exported and used for inference or serving.

    Args:
        path: path to export or load generator
        is_regression: whether the task is a regression task
    """

    def __init__(self,
                 path=None,
                 is_regression=False,
                 is_binary_classification=False):
        super().__init__(path)
        self._is_regression = is_regression
        self._is_binary_classification = is_binary_classification

        self._model = None

    def build_from_model(self, model):
        """
        Build generator from model

        Args:
            model (paragen.models.AbstractModel): a neural model
        """
        self._model = model

    def _forward(self, *args):
        """
        Infer a sample as model in evaluation mode, and predict results from logits predicted by model

        Args:
            inputs: inference inputs
        """
        output = self._model(*args)
        if not self._is_regression:
            if self._is_binary_classification:
                output = (output > 0.5).long()
            else:
                _, output = output.max(dim=-1)
        return output

    def reset(self, mode):
        """
        Reset generator states.

        Args:
            mode: running mode
        """
        if mode != 'train':
            self.eval()
        self._mode = mode
        self._model.reset(mode)

    @property
    def model(self):
        return self._model

    @property
    def input_slots(self):
        """
        Generator input slots that is auto-detected
        """
        return inspect_fn(self._model.forward)
