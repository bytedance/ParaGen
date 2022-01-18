from typing import Dict
import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn

from paragen.generators import AbstractGenerator, register_generator
from paragen.modules.utils import create_padding_mask_from_length
from paragen.utils.io import mkdir, UniIO
from paragen.utils.ops import local_seed
from paragen.utils.runtime import Environment
from paragen.utils.tensor import to_device


@register_generator
class GLATAuxGenerator(AbstractGenerator):
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
                 search: Dict=None,
                 path=None):
        super().__init__(path)
        self._search_configs = search

        self._model = None
        self._encoder, self._length_predictor, self._calc_decoder_input, self._decoder = None, None, None, None
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
        self._length_predictor, self._calc_decoder_input = model.length_predictor, model.calc_decoder_input
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
        if tgt_padding_mask is None:
            tgt_length = self._length_predictor(length_token)
            tgt_length = tgt_length.max(dim=-1).indices
            tgt_padding_mask = create_padding_mask_from_length(tgt_length)
        decoder_input = self._calc_decoder_input(src_padding_mask,
                                                 tgt_padding_mask)
        with local_seed(self.seed):
            logits = self._decoder(decoder_input[0], src_hidden, tgt_padding_mask, src_padding_mask)
        _, decoder_output = logits.max(dim=-1)
        decoder_output = decoder_output.masked_fill_(tgt_padding_mask, self._tgt_special_tokens['pad'])
        return decoder_output

    def export(self, path, net_input, **kwargs):
        """
        Export self to `path` by export model directly

        Args:
            path: path to store serialized model
            net_input: fake net_input for tracing the model
        """
        self.eval()
        self.reset('infer')
        env = Environment()
        with torch.no_grad():
            logger.info('trace GLATGenerator')
            net_input = to_device(net_input, device=env.device)
            # traced_model = torch.jit.trace_module(self, {'forward': (net_input['src'],)})
            encoder = Encoder(self._encoder, self._length_predictor)
            traced_encoder = torch.jit.trace_module(encoder, {'forward': (net_input['src'],)})
            src_hidden, src_padding_mask, tgt_length = traced_encoder(net_input['src'])
            tgt_padding_mask = create_padding_mask_from_length(tgt_length)
            decoder = Decoder(self._calc_decoder_input, self._decoder, self._tgt_special_tokens['pad'])
            traced_decoder = torch.jit.trace_module(decoder, {'forward': (tgt_padding_mask, src_padding_mask, src_hidden)})
        logger.info(f'save glat to {path}/encoder.pt & {path}/decoder.pt')
        mkdir(path)
        with UniIO(f'{path}/encoder.pt', 'wb') as fout:
            torch.jit.save(traced_encoder, fout)
        with UniIO(f'{path}/decoder.pt', 'wb') as fout:
            torch.jit.save(traced_decoder, fout)
        if kwargs['use_onnx']:
            logger.info('export glat to {}/model.onnx'.format(path))
            torch.onnx.export(self,
                (net_input['src'],),
                '{}/model.onnx'.format(path),
                strip_doc_string=True,
                do_constant_folding=True,
                opset_version=11
            )

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

    def __init__(self, encoder, length_predictor):
        super().__init__()
        self._encoder, self._length_predictor = encoder, length_predictor

    def forward(self, src):
        src_hidden, src_padding_mask, length_token = self._encoder(src)
        tgt_length = self._length_predictor(length_token)
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
