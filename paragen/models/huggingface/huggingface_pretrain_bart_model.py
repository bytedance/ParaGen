from transformers import BartConfig, BartForConditionalGeneration

from paragen.models import register_model
from paragen.models.abstract_model import AbstractModel


@register_model
class HuggingfacePretrainBartModel(AbstractModel):
    """
    HuggingfacePretrainBartModel is a pretrained bart model built on
    huggingface pretrained bart models.
    """

    def __init__(self):
        super().__init__()

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
        self._config = BartConfig(vocab_size=vocab_size, pad_token_id=special_tokens['pad'])
        self._model = BartForConditionalGeneration(self._config)
        self._special_tokens = special_tokens

    def forward(self, enc_input, dec_input):
        """
        Compute output with neural input

        Args:
            enc_input: encoder input sequence
            dec_input: decoder input sequence

        Returns:
            - log probability of next tokens in sequences
        """
        output = self._model(enc_input,
                             attention_mask=enc_input.ne(self._special_tokens['pad']),
                             decoder_input_ids=dec_input,
                             use_cache=self._mode == 'infer')
        output = output[0]
        return output
