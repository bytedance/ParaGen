import importlib
import os

from paragen.utils.registry import setup_registry

from .abstract_tokenizer import AbstractTokenizer

register_tokenizer, create_tokenizer, registry = setup_registry('tokenizer', AbstractTokenizer)

modules_dir = os.path.dirname(__file__)
for file in os.listdir(modules_dir):
    path = os.path.join(modules_dir, file)
    if (
        not file.startswith('_')
        and not file.startswith('.')
        and (file.endswith('.py') or os.path.isdir(path))
    ):
        module_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module('paragen.tokenizers.' + module_name)

