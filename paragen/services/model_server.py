from typing import Dict
import pickle
import logging
logger = logging.getLogger(__name__)

import torch

from paragen.utils.ops import auto_map_args
from paragen.utils.runtime import Environment
from paragen.utils.tensor import to_device


class ModelServer:
    """
    ModelServer is a thrift server running neural model at backend.

    Args:
        generator: neural inference model
    """

    def __init__(self, generator):
        self._generator = generator
        self._env = Environment()

        self._generator.eval()

    def infer(self, net_input):
        """
        Inference with neural model.

        Args:
            net_input: neural model

        Returns:
            - neural output
        """
        try:
            net_input = pickle.loads(net_input)
            if isinstance(net_input, Dict):
                net_input = auto_map_args(net_input, self._generator.input_slots)
            net_input = to_device(net_input, self._env.device, fp16=self._env.fp16)
            with torch.no_grad():
                self._generator.reset('infer')
                net_output = self._generator(*net_input)
            net_output = to_device(net_output, 'cpu')
            net_output = pickle.dumps(net_output)
            return net_output
        except Exception as e:
            logger.warning(str(e))
            return None
