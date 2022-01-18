import importlib
import os

from paragen.utils.registry import setup_registry

from .abstract_model import AbstractModel

register_model, create_model, registry = setup_registry('model', AbstractModel)

models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if (
        not file.startswith('_')
        and not file.startswith('.')
        and (file.endswith('.py') or os.path.isdir(path))
    ):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module('paragen.models.' + model_name)
