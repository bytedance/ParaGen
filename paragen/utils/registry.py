import json
import logging
logger = logging.getLogger(__name__)

from paragen.utils.data import possible_eval
from paragen.utils.io import jsonable
from paragen.utils.ops import deepcopy_on_ref

MODULE_REGISTRY = {}


def setup_registry(registry, base_cls, force_extend=True):
    """
    Set up registry for a certain class

    Args:
        registry: registry name
        base_cls: base class of a certain class
        force_extend: force a new class extend the base class

    Returns:
        - decorator to register a subclass
        - function to create a subclass with configurations
        - registry dictionary
    """

    if registry not in  MODULE_REGISTRY:
        MODULE_REGISTRY[registry] = {}

    def register_cls(cls):
        """
        Register a class with its name

        Args:
            cls: a new class fro registration
        """
        name = cls.__name__.lower()
        if name in MODULE_REGISTRY[registry]:
            raise ValueError('Cannot register duplicate {} class ({})'.format(registry, name))
        if force_extend and not issubclass(cls, base_cls):
            raise ValueError('Class {} must extend {}'.format(name, base_cls.__name__))
        if name in MODULE_REGISTRY[registry]:
            raise ValueError('Cannot register class with duplicate class name ({})'.format(name))
        MODULE_REGISTRY[registry][name] = cls
        return cls

    def create_cls(configs=None):
        """
        Create a class with configuration

        Args:
            configs: configuration dictionary for building class

        Returns:
            - an instance of class
        """
        configs = deepcopy_on_ref(configs)
        name = configs.pop('class')
        json_configs = {k: v for k, v in configs.items() if jsonable(k) and jsonable(v)}
        logger.info('Creating {} class with configs \n{}\n'.format(name, json.dumps(json_configs, indent=4, sort_keys=True)))
        assert name.lower() in MODULE_REGISTRY[registry], f"{name} is not implemented in ParaGen"
        cls = MODULE_REGISTRY[registry][name.lower()]
        kwargs = {}
        for k, v in configs.items():
            kwargs[k] = possible_eval(v)
        return cls(**kwargs)

    return register_cls, create_cls, MODULE_REGISTRY[registry]
