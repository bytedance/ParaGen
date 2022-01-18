import importlib
import os

from paragen.utils.registry import setup_registry
from paragen.utils.runtime import Environment

from .abstract_sampler import AbstractSampler
from .distributed_sampler import DistributedSampler

register_sampler, _create_sampler, registry = setup_registry('sampler', AbstractSampler)


def create_sampler(configs, is_training=False):
    """
    Create a sampler.
    Note in distributed training, sampler should be further wrapped with a DistributedSampler.

    Args:
        configs: sampler configuration
        is_training: whether the sampler is used for training.

    Returns:
        a data sampler
    """
    sampler = _create_sampler(configs)
    env = Environment()
    if env.distributed_world > 1 and is_training:
        sampler = DistributedSampler(sampler)
    return sampler


modules_dir = os.path.dirname(__file__)
for file in os.listdir(modules_dir):
    path = os.path.join(modules_dir, file)
    if (
        not file.startswith('_')
        and not file.startswith('.')
        and (file.endswith('.py') or os.path.isdir(path))
    ):
        module_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module('paragen.samplers.' + module_name)
