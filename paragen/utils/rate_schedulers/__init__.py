import importlib
import os

from paragen.utils.registry import setup_registry

from .abstract_rate_scheduler import AbstractRateScheduler

register_rate_scheduler, _create_rate_scheduler, registry = setup_registry('rate_scheduler', AbstractRateScheduler)


def create_rate_scheduler(configs):
    if isinstance(configs, float):
        configs = {'class': 'ConstantRateScheduler', 'rate': configs}
    rate_schduler = _create_rate_scheduler(configs)
    return rate_schduler


modules_dir = os.path.dirname(__file__)
for file in os.listdir(modules_dir):
    path = os.path.join(modules_dir, file)
    if (
        not file.startswith('_')
        and not file.startswith('.')
        and (file.endswith('.py') or os.path.isdir(path))
    ):
        module_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module('paragen.utils.rate_schedulers.' + module_name)


