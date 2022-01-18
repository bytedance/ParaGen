from transformers import AutoConfig, AutoModelForSequenceClassification

from paragen.models import register_model
from paragen.models.abstract_model import AbstractModel


@register_model
class HuggingfaceSequenceClassificationModel(AbstractModel):
    """
    HuggingfaceSequenceClassificationModel is a sequence classification architecture built on
    huggingface sequence classification models.

    Args:
        pretrained_model: pretrained_model in huggingface
        num_labels: number of labels
    """

    def __init__(self, pretrained_model, num_labels=2):
        super().__init__()
        self._pretrained_model = pretrained_model
        self._num_labels = num_labels

        self._config = None
        self._model = None
        self._special_tokens = None

    def _build(self, vocab_size, special_tokens):
        """
        Build model with vocabulary size and special tokens

        Args:
            vocab_size: vocabulary size of input sequence
            special_tokens: special tokens of input sequence
        """
        self._config = AutoConfig.from_pretrained(
            self._pretrained_model,
            num_labels=self._num_labels
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._pretrained_model,
            config=self._config,
        )
        self._special_tokens = special_tokens

    def forward(self, input):
        """
        Compute output with neural input

        Args:
            input: input source sequences

        Returns:
            - log probability of labels
        """
        output = self._model(input, attention_mask=input.ne(self._special_tokens['pad']))
        output = output.logits if self._num_labels > 1 else output.logits.squeeze(dim=-1)
        return output
