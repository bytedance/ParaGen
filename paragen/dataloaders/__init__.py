import importlib
import os

from paragen.utils.ops import deepcopy_on_ref
from paragen.utils.registry import setup_registry

from .abstract_dataLoader import AbstractDataLoader

register_dataloader, _build_dataloader, registry = setup_registry('dataloader', AbstractDataLoader)


def build_dataloader(configs, dataset, sampler=None, collate_fn=None, post_collate=False):
    """
    Build a dataloader

    Args:
        configs: dataloader configs
        dataset: dataset storing samples
        sampler: sample strategy
        collate_fn: collate function during data fetching with torch.utils.data.DataLoader
        post_collate: whether to perform collate_fn after data fetching

    Returns:
        AbstractDataLoader
    """
    configs = deepcopy_on_ref(configs)
    configs.update({
        'dataset': dataset,
        'collate_fn': collate_fn if not post_collate else None,
        'post_collate_fn': collate_fn if post_collate else None
    })
    if sampler is not None:
        configs['sampler'] = sampler
    dataloader = _build_dataloader(configs)
    return dataloader


modules_dir = os.path.dirname(__file__)
for file in os.listdir(modules_dir):
    path = os.path.join(modules_dir, file)
    if (
        not file.startswith('_')
        and not file.startswith('.')
        and (file.endswith('.py') or os.path.isdir(path))
    ):
        module_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module('paragen.dataloaders.' + module_name)
