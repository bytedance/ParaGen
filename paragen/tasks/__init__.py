import importlib
import os

from paragen.utils.registry import setup_registry

TRAIN, VALID, EVALUATE, SERVE = 'TRAIN', 'VALID', 'EVALUATE', 'SERVE'

from .abstract_task import AbstractTask

register_task, create_task, registry = setup_registry('task', AbstractTask)

modules_dir = os.path.dirname(__file__)
for file in os.listdir(modules_dir):
    path = os.path.join(modules_dir, file)
    if (
        not file.startswith('_')
        and not file.startswith('.')
        and (file.endswith('.py') or os.path.isdir(path))
    ):
        module_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module('paragen.tasks.' + module_name)

