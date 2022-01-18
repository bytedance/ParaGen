from transformers import MBartConfig, MBartForConditionalGeneration

from paragen.models import register_model
from paragen.models.abstract_model import AbstractModel


@register_model
class HuggingfacePretrainMBartModel(AbstractModel):
    """
    HuggingfacePretrainBartModel is a pretrained bart model built on
    huggingface pretrained bart models.
    """

    def __init__(self, path=None, pretrained_path=None):
        super().__init__(path)

        self._config = None
        self._model = None
        self._special_tokens = None
        self._pretrained_path = pretrained_path

    def _build(self, vocab_size, special_tokens):
        """
        Build model with vocabulary size and special tokens

        Args:
            vocab_size: vocabulary size of input sequence
            special_tokens: special tokens of input sequence
        """
        self._config = MBartConfig(vocab_size=vocab_size, pad_token_id=special_tokens['pad'])
        if self._pretrained_path:
            self._model = MBartForConditionalGeneration(self._config).from_pretrained(self._pretrained_path)
        else:
            self._model = MBartForConditionalGeneration(self._config)
        self._special_tokens = special_tokens

    def forward(self, src, tgt):
        """
        Compute output with neural input

        Args:
            src: encoder input sequence
            tgt: decoder input sequence

        Returns:
            - log probability of next tokens in sequences
        """
        output = self._model(src,
                             attention_mask=src.ne(self._special_tokens['pad']),
                             decoder_input_ids=tgt,
                             use_cache=self._mode == 'infer')
        output = output[0]
        return output
    
    def generate(self, src, tgt_langtok_id, max_length, beam):
        return self._model.generate(input_ids=src, decoder_start_token_id=tgt_langtok_id, max_length=max_length, num_beams=beam)
