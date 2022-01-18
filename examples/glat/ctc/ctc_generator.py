from typing import Dict

import torch
import torch.nn as nn

from paragen.generators import AbstractGenerator, register_generator
from paragen.utils.ops import local_seed


@register_generator
class CTCGenerator(AbstractGenerator):
    """
    SequenceGenerator is combination of a model and search algorithm.
    It processes in a multi-step fashion while model processes only one step.
    It is usually separated into encoder and search with decoder, and is
    exported and load with encoder and search module.

    Args:
        search: search configs
        path: path to export or load generator
    """

    def __init__(self,
                 search: Dict = None,
                 path=None):
        super().__init__(path)
        self._search_configs = search

        self._model = None
        self._encoder, self._calc_decoder_input, self._decoder = None, None, None
        self._src_special_tokens, self._tgt_special_tokens = None, None
        self._seed = None

    def build_from_model(self, model, src_special_tokens, tgt_special_tokens):
        """
        Build generator from model and search.

        Args:
            model (paragen.models.EncoderDecoder): an encoder-decoder model to be wrapped
            src_special_tokens: special tokens in source vocabulary
            tgt_special_tokens: special tokens in target vocabulary
        """
        self._model = model
        self._encoder, self._decoder = model.encoder, model.decoder
        self._calc_decoder_input = model.calc_decoder_input
        self._src_special_tokens, self._tgt_special_tokens = src_special_tokens, tgt_special_tokens

    def _forward(self, src, tgt_padding_mask=None):
        """
        Infer a sample as model in evaluation mode.
        Compute encoder output first and decode results with search module

        Args:
            src: encoder inputs
            tgt_padding_mask: decoder inputs

        Returns:
            decoder_output: results inferred by search algorithm on decoder
        """
        src_hidden, src_padding_mask, length_token = self._encoder(src)

        decoder_input = self._calc_decoder_input(src_padding_mask,
                                                 tgt_padding_mask)
        with local_seed(self.seed):
            logits = self._decoder(decoder_input, src_hidden, tgt_padding_mask, src_padding_mask)
        _, decoder_output = logits.max(dim=-1)
        decoder_output = decoder_output.masked_fill_(tgt_padding_mask, self._tgt_special_tokens['pad'])
        blank_id = self._tgt_special_tokens['unk']
        pad_id = self._tgt_special_tokens['pad']
        bsz, seq_len = decoder_output.size()
        new_decoder_output = torch.zeros_like(decoder_output).fill_(pad_id)
        for bidx in range(bsz):
            cidx = 0
            for sidx in range(0, seq_len - 1):
                if decoder_output[bidx, sidx] == decoder_output[bidx, sidx + 1] or decoder_output[bidx, sidx] == blank_id:
                    continue
                else:
                    new_decoder_output[bidx, cidx] = decoder_output[bidx, sidx]
                    cidx += 1
            if decoder_output[bidx, -1] != blank_id:
                new_decoder_output[bidx, cidx] = decoder_output[bidx, -1]

        return new_decoder_output

    def reset(self, mode):
        """
        Reset generator states.

        Args:
            mode: running mode
        """
        self.eval()
        if self._traced_model is None:
            self._encoder.reset(mode)
            self._decoder.reset(mode)

    @property
    def seed(self):
        return self._model.seed


class Encoder(nn.Module):

    def __init__(self, encoder):
        super().__init__()
        self._encoder = encoder

    def forward(self, src):
        src_hidden, src_padding_mask, length_token = self._encoder(src)
        tgt_length = (~src_padding_mask).sum(-1) * 2
        tgt_length = tgt_length.max(dim=-1).indices
        return src_hidden, src_padding_mask, tgt_length


class Decoder(nn.Module):

    def __init__(self, calc_decoder_input, decoder, pad):
        super().__init__()
        self._calc_decoder_input, self._decoder = calc_decoder_input, decoder
        self._pad = pad

    def forward(self, tgt_padding_mask, src_padding_mask, src_hidden):
        decoder_input = self._calc_decoder_input(src_padding_mask,
                                                 tgt_padding_mask)
        logits = self._decoder(decoder_input, src_hidden, tgt_padding_mask, src_padding_mask)
        _, decoder_output = logits.max(dim=-1)
        decoder_output = decoder_output.masked_fill_(tgt_padding_mask, self._pad)
        return decoder_output
