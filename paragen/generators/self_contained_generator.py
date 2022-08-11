from paragen.generators import AbstractGenerator, register_generator


@register_generator
class SelfContainedGenerator(AbstractGenerator):
    """
    SelfContainedGenerator use self-implemented generate function within model.
    Generator has the same function and interface as model.
    It can be directly exported and used for inference or serving.

    Args:
        path: path to export or load generator
    """

    def __init__(self,
                 path=None, **kwargs):
        super().__init__(path)
        self._kwargs = kwargs
        self._model = None

    def build_from_model(self, model, *args, **kwargs):
        """
        Build generator from model

        Args:
            model (paragen.models.AbstractModel): a neural model
        """
        self._model = model

    def _forward(self, *args, **kwargs):
        """
        Infer a sample as model in evaluation mode, and predict results from logits predicted by model
        """
        kwargs.update(self._kwargs)
        output = self._model.generate(*args, **kwargs)
        return output

    @property
    def model(self):
        return self._model

    def reset(self, mode):
        """
        Reset generator states.

        Args:
            mode: running mode
        """
        self.eval()
        self._mode = mode
        self._model.reset(mode)
