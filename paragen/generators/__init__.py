import importlib
import os

from paragen.utils.registry import setup_registry

from .abstract_generator import AbstractGenerator

register_generator, create_generator, registry = setup_registry('generator', AbstractGenerator)

modules_dir = os.path.dirname(__file__)
for file in os.listdir(modules_dir):
    path = os.path.join(modules_dir, file)
    if (
        not file.startswith('_')
        and not file.startswith('.')
        and (file.endswith('.py') or os.path.isdir(path))
    ):
        module_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module('paragen.generators.' + module_name)

