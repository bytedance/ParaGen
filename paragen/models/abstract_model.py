import logging
logger = logging.getLogger(__name__)

import torch.nn as nn

from paragen.utils.runtime import Environment as E
from paragen.utils.tensor import save_ckpt
from paragen.utils.tensor import get_avg_ckpt


class AbstractModel(nn.Module):
    """
    AbstractModel is abstract class for models defining inferfaces.

    Args:
        path: path to restore checkpoints
    """

    def __init__(self, path=None):
        super().__init__()
        self._path = path

        self._mode = 'train'
        self._states = {}

    def build(self, *args, **kwargs):
        """
        Build neural model with task instances.
        It wraps `_build` function with restoring and moving to cuda.
        """
        self._build(*args, **kwargs)
        logger.info('neural network architecture\n{}'.format([_ for _ in self.children()]))
        logger.info('parameter size: {}'.format(sum(p.numel() for p in self.parameters())))

        e = E()
        if self._path is not None:
            logger.info(f'load model from {self._path}')
            self.load(self._path, e.device)

        if e.device.startswith('cuda'):
            logger.info('move model to {}'.format(e.device))
            self.cuda(e.device)

    def _build(self, *args, **kwargs):
        """
        Build neural model with task instances.
        """
        raise NotImplementedError

    def forward(self, *input):
        """
        Compute output with neural input

        Args:
            *input: neural inputs
        """
        raise NotImplementedError

    def load(self, path, device, strict=False):
        """
        Load model from path and move model to device.

        Args:
            path: path to restore model
            device: running device
            strict: load model strictly
        """
        paths = path.split(',')
        state_dict = get_avg_ckpt(paths, device=device)

        if 'model' in state_dict:
            state_dict = state_dict['model']
        state_dict = {key.lstrip('module.'): val for key, val in state_dict.items()}
        mismatched = self.load_state_dict(state_dict, strict=strict)

        if not strict:
            logger.info("keys IN this model but NOT IN loaded model >>> ")
            if len(mismatched.missing_keys) > 0:
                for ele in mismatched.missing_keys:
                    logger.info(f"    - {ele}")
            else:
                logger.info("    - None")
            logger.info("keys NOT IN this model but IN loaded model >>> ")
            if len(mismatched.unexpected_keys) > 0:
                for ele in mismatched.unexpected_keys:
                    logger.info(f"    - {ele}")
            else:
                logger.info("    - None")

    def save(self, path):
        """
        Save model to path.

        Args:
            path: path to save model
        """
        save_ckpt({'model': self.state_dict()}, path)

    def update_states(self, *args, **kwargs):
        """
        Update internal networks states.
        """
        raise NotImplementedError

    @property
    def states(self):
        return self._states

    def reset(self, *args, **kwargs):
        """
        Reset neural model states.
        """
        pass

    def is_pretrained(self):
        return self._path is not None
