import importlib
import os

from paragen.utils.registry import setup_registry

from .abstract_search import AbstractSearch

register_search, create_search, registry = setup_registry('search', AbstractSearch)

modules_dir = os.path.dirname(__file__)
for file in os.listdir(modules_dir):
    path = os.path.join(modules_dir, file)
    if (
        not file.startswith('_')
        and not file.startswith('.')
        and (file.endswith('.py') or os.path.isdir(path))
    ):
        module_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module('paragen.modules.search.' + module_name)

