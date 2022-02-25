import torchvision.models as models

from paragen.models import AbstractModel, register_model


@register_model
class TorchVisionModel(AbstractModel):
    """
    SequenceClassificationModel is a general sequence classification architecture consisting of
    one encoder and one classifier.
    Args:
        encoder: encoder configuration
        labels: number of labels
        path: path to restore model
    """

    def __init__(self,
                 model,
                 num_classes,
                 path=None,
                 **kwargs):
        super().__init__(path)
        self._model_name = model
        self._num_classes = num_classes
        self._path = path
        self._kwargs = kwargs

        self._model = None

    def _build(self):
        """
        Build model with vocabulary size and special tokens
        """
        cls = getattr(models, self._model_name)
        self._model = cls(num_classes=self._num_classes, **self._kwargs)

    def forward(self, input):
        """
        Compute output with neural input
        Args:
            input: input image batch
        Returns:
            - log probability of labels
        """
        return self._model(input)
