import logging
logger = logging.getLogger(__name__)

import torch
from transformers import BartConfig, BartForConditionalGeneration

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

    def __init__(
        self,
        arch=None,
        max_position=None,
        pretrained_model=None,
        path=None
    ):
        super().__init__(path=path)
        self._arch = arch
        self._max_position = max_position
        self._pretrained_model = pretrained_model
        if self._pretrained_model is not None:
            if self._arch is None:
                self._arch = self._pretrained_model
            else:
                assert self._arch == self._pretrained_model
            assert max_position is None
        assert self._arch is not None

        self._config = None
        self._model = None
        self._special_tokens = None

    def _build(self, src_vocab_size, tgt_vocab_size, src_special_tokens, tgt_special_tokens):
        """
        Build model with vocabulary size and special tokens

        Args:
            vocab_size: vocabulary size of input sequence
            special_tokens: special tokens of input sequence
        """
        assert self._pretrained_model is not None or \
               (src_vocab_size is not None and src_special_tokens is not None and tgt_vocab_size is not None and tgt_special_tokens is not None)
        assert src_vocab_size == tgt_vocab_size
        assert src_special_tokens == tgt_special_tokens
        assert src_special_tokens['bos'] == 0

        self._config = BartConfig.from_pretrained(self._arch)
        if self._pretrained_model is not None:
            self._model = BartForConditionalGeneration.from_pretrained(self._pretrained_model, forced_bos_token_id=0)
        else:
            self._config.vocab_size = src_vocab_size
            self._config.pad_token_id, self._config.bos_token_id, self._config.eos_token_id = src_special_tokens['pad'], src_special_tokens['bos'], src_special_tokens['eos']
            if self._max_position is not None:
                self._config.max_position_embeddings = self._max_position
            self._model = BartForConditionalGeneration(self._config)
        self._special_tokens = src_special_tokens


    def load(self, path, device, strict=False):
        """
        Load model from path and move model to device.

        Args:
            path: path to restore model
            device: running device
            strict: load model strictly
        """
        load_huggingface_model = False
        if path.startswith('hf:'):
            path = path[3:]
            load_huggingface_model = True
        if load_huggingface_model:
            with open(path, 'rb') as fin:
                state_dict = torch.load(fin, map_location='cpu')
                self._model.load_state_dict(state_dict)
        else:
            super().load(path, device=device, strict=strict)

    def forward(self, src, tgt):
        """
        Compute output with neural input

        Args:
            input: input sequence

        Returns:
            - log probability of start and end position
        """
        output = self._model(src,
                             attention_mask=src.ne(self._special_tokens['pad']),
                             decoder_input_ids=tgt,
                             decoder_attention_mask=tgt.ne(self._special_tokens['pad']),)
        return output.logits

    def generate(self, *args, **kwargs):
        return self._model.generate(*args, **kwargs)
