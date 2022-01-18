from transformers import AutoConfig, AutoModelForQuestionAnswering

from paragen.models import register_model
from paragen.models.abstract_encoder_decoder_model import AbstractEncoderDecoderModel
from paragen.modules.layers.classifier import HuggingfaceClassifier


@register_model
class HuggingfaceExtractiveQuestionAnsweringModel(AbstractEncoderDecoderModel):
    """
    HuggingfaceExtractiveQuestionAnsweringModel is a extractive question answering model built on
    huggingface extractive question answering models.

    Args:
        pretrained_model: pretrained_model in huggingface
        has_answerable: has answerable problem
        path: path to restore model
    """

    def __init__(self, pretrained_model, has_answerable=False, path=None):
        super().__init__(path=path)
        self._pretrained_model = pretrained_model
        self._has_answerable = has_answerable

        self._config = None
        self._model = None
        self._special_tokens = None
        self._encoder, self._decoder = None, None
        if self._has_answerable:
            self._classification_head = None

    def _build(self, vocab_size, special_tokens):
        """
        Build model with vocabulary size and special tokens

        Args:
            vocab_size: vocabulary size of input sequence
            special_tokens: special tokens of input sequence
        """
        self._config = AutoConfig.from_pretrained(self._pretrained_model)
        self._model = AutoModelForQuestionAnswering.from_pretrained(self._pretrained_model, config=self._config,)
        self._special_tokens = special_tokens

        if self._has_answerable:
            self._classification_head = HuggingfaceClassifier(self._model.d_model, 2)

    def forward(self, input, answerable=None, start_positions=None, end_positions=None):
        """
        Compute output with neural input

        Args:
            input: input sequence
            answerable: gold answerable
            start_positions: gold start position
            end_positions: gold end position

        Returns:
            - log probability of start and end position
        """
        output = self._model(input,
                             attention_mask=input.ne(self._special_tokens['pad']),
                             start_positions=start_positions,
                             end_positions=end_positions)
        return output

    def loss(self, input, answerable=None, start_positions=None, end_positions=None):
        """
        Compute loss from network inputs

        Args:
            input: input sequence
            answerable: gold answerable
            start_positions: gold start position
            end_positions: gold end position

        Returns:
            - loss
        """
        output = self(input, answerable, start_positions, end_positions)
        return output[0]

