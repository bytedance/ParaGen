from transformers import BartForConditionalGeneration

from paragen.models import register_model
from paragen.models.abstract_encoder_decoder_model import AbstractEncoderDecoderModel


@register_model
class HuggingfaceBartModel(AbstractEncoderDecoderModel):
    """
    HuggingfaceExtractiveQuestionAnsweringModel is a extractive question answering model built on
    huggingface extractive question answering models.

    Args:
        pretrained_model: pretrained_model in huggingface
        has_answerable: has answerable problem
        path: path to restore model
    """

    def __init__(self, pretrained_model, path=None):
        super().__init__(path=path)
        self._pretrained_model = pretrained_model

        self._config = None
        self._model = None
        self._special_tokens = None

    def build(self, src_vocab_size, tgt_vocab_size, src_special_tokens, tgt_special_tokens):
        """
        Build model with vocabulary size and special tokens

        Args:
            vocab_size: vocabulary size of input sequence
            special_tokens: special tokens of input sequence
        """
        assert src_vocab_size == tgt_vocab_size
        assert src_special_tokens == tgt_special_tokens
        assert src_special_tokens['bos'] == 0
        self._model = BartForConditionalGeneration.from_pretrained(self._pretrained_model, forced_bos_token_id=0)
        self._special_tokens = src_special_tokens

    def forward(self, src, tgt):
        """
        Compute output with neural input

        Args:
            input: input sequence

        Returns:
            - log probability of start and end position
        """
        output = self._model(src,
                             attention_mask=src.eq(self._special_tokens['pad']),
                             decoder_input_ids=tgt,
                             decoder_attention_mask=tgt.eq(self._special_tokens['pad']),)
        return output.logits

    def generate(self, *args, **kwargs):
        return self._model.generate(*args, **kwargs)
