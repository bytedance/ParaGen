import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn

from paragen.utils.ops import inspect_fn
from paragen.utils.runtime import Environment
from paragen.utils.io import UniIO, mkdir


class AbstractGenerator(nn.Module):
    """
    AbstractGenerator wrap a model with inference algorithms.
    It can be directly exported and used for inference or serving.

    Args:
        path: path to restore traced model
    """

    def __init__(self, path):
        super().__init__()

        self._path = path
        self._traced_model = None
        self._model = None
        self._mode = 'infer'

    def build(self, *args, **kwargs):
        """
        Build or load a generator
        """
        if self._path is not None:
            self.load()
        else:
            self.build_from_model(*args, **kwargs)

        self._env = Environment()
        if self._env.device.startswith('cuda'):
            logger.info('move model to {}'.format(self._env.device))
            self.cuda(self._env.device)

    def build_from_model(self, *args, **kwargs):
        """
        Build generator from model
        """
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """
        Infer a sample in evaluation mode.
        We auto detect whether the inference model is traced, and use appropriate model to perform inference.
        """
        if self._traced_model is not None:
            return self._traced_model(*args, **kwargs)
        else:
            return self._forward(*args, **kwargs)

    def _forward(self, *args, **kwargs):
        """
        Infer a sample in evaluation mode with torch model.
        """
        raise NotImplementedError

    def export(self, path, net_input, **kwargs):
        """
        Export self to `path` by export model directly

        Args:
            path: path to store serialized model
            net_input: fake net_input for tracing the model
        """
        self.eval()
        with torch.no_grad():
            logger.info('trace model {}'.format(self._model.__class__.__name__))
            model = torch.jit.trace_module(self._model, {'forward': net_input})
        mkdir(path)
        logger.info('save model to {}/model'.format(path))
        with UniIO('{}/model'.format(path), 'wb') as fout:
            torch.jit.save(model, fout)

    def load(self):
        """
        Load a serialized model from path
        """
        logger.info('load model from {}'.format(self._path))
        with UniIO(self._path, 'rb') as fin:
            self._traced_model = torch.jit.load(fin)

    def reset(self, *args, **kwargs):
        """
        Reset generator states.
        """
        pass

    @property
    def input_slots(self):
        """
        Generator input slots that is auto-detected
        """
        return inspect_fn(self._forward)
