import importlib
import os

from paragen.utils.registry import setup_registry

from .abstract_evaluator import AbstractEvaluator

register_evaluator, create_evaluator, registry = setup_registry('evaluator', AbstractEvaluator)

modules_dir = os.path.dirname(__file__)
for file in os.listdir(modules_dir):
    path = os.path.join(modules_dir, file)
    if (
        not file.startswith('_')
        and not file.startswith('.')
        and (file.endswith('.py') or os.path.isdir(path))
    ):
        module_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module('paragen.evaluators.' + module_name)
