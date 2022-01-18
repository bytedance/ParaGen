from typing import Dict
import logging
logger = logging.getLogger(__name__)

import torch

from paragen.generators import AbstractGenerator, register_generator
from paragen.modules.encoders import AbstractEncoder
from paragen.modules.search import create_search, AbstractSearch
from paragen.utils.runtime import Environment
from paragen.utils.io import UniIO, mkdir, cp
from paragen.utils.tensor import to_device


@register_generator
class SequenceGenerator(AbstractGenerator):
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
        self._encoder, self._search = None, None
        self._src_special_tokens, self._tgt_special_tokens = None, None
        self._env = None

    def build_from_model(self, model, src_special_tokens, tgt_special_tokens):
        """
        Build generator from model and search.

        Args:
            model (paragen.models.EncoderDecoder): an encoder-decoder model to be wrapped
            src_special_tokens (dict): source special token dict
            tgt_special_tokens (dict): target special token dict
        """
        self._model = model
        self._encoder = model.encoder
        self._src_special_tokens, self._tgt_special_tokens = src_special_tokens, tgt_special_tokens

        self._search = create_search(self._search_configs)
        self._search.build(decoder=model.decoder,
                           bos=self._tgt_special_tokens['bos'],
                           eos=self._tgt_special_tokens['eos'],
                           pad=self._tgt_special_tokens['pad'])
        self._env = Environment()

    def _forward(self, encoder, decoder, search=None):
        """
        Infer a sample as model in evaluation mode.
        Compute encoder output first and decode results with search module

        Args:
            encoder (tuple): encoder inputs
            decoder (tuple): decoder inputs
            search (tuple): search states

        Returns:
            decoder_output: results inferred by search algorithm on decoder
        """
        if not search:
            search = tuple()
        encoder_output = self._encoder(*encoder)
        _, decoder_output = self._search(*decoder, *encoder_output, *search)
        return decoder_output

    def export(self, path, net_input, *args, **kwargs):
        """
        Export self to `path` by export model directly

        Args:
            path: path to store serialized model
            net_input: fake net_input for tracing the model
        """
        self.eval()
        self.reset('infer')
        net_input = to_device(net_input, device=self._env.device)
        with torch.no_grad():
            logger.info(f'trace encoder {self._encoder.__class__.__name__}')
            encoder = torch.jit.trace_module(self._encoder, {'forward': net_input['encoder']})
            logger.info(f'script search {self._search.__class__.__name__}')
            search = torch.jit.script(self._search)
        mkdir(path)
        logger.info(f'save encoder to {path}/encoder')
        with UniIO(f'{path}/encoder', 'wb') as fout:
            torch.jit.save(encoder, fout)
        logger.info(f'save search to {path}/search')
        with UniIO(f'{path}/search', 'wb') as fout:
            torch.jit.save(search, fout)
        if 'use_onnx' in kwargs and kwargs['use_onnx']:
            logger.info('exporting onnx model')
            torch.onnx.export(self,
                              (net_input['encoder'], net_input['decoder']),
                              'model.onnx',
                              strip_doc_string=True,
                              do_constant_folding=True,
                              opset_version=11,
                              )
            cp('model.onnx', f'{path}/model.onnx')

    def load(self):
        """
        Load generator (encoder & search) from path
        """
        logger.info('load encoder from {}/encoder'.format(self._path))
        with UniIO('{}/encoder'.format(self._path), 'rb') as fin:
            self._encoder = torch.jit.load(fin)
        logger.info('load search from {}/search'.format(self._path))
        with UniIO('{}/search'.format(self._path), 'rb') as fin:
            self._search = torch.jit.load(fin)

    @property
    def encoder(self):
        return self._encoder

    @property
    def search(self):
        return self._search

    def reset(self, mode):
        """
        Reset generator states.

        Args:
            mode: running mode
        """
        self.eval()
        self._mode = mode
        if self._traced_model is None:
            if isinstance(self._encoder, AbstractEncoder):
                self._encoder.reset(mode)
            if isinstance(self._search, AbstractSearch):
                self._search.reset(mode)
            if self._env.device == 'cuda':
                torch.cuda.empty_cache()
