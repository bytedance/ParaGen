import importlib
import json
import logging
logger = logging.getLogger(__name__)

from tqdm import tqdm
import torch
from functools import wraps


def singleton(cls):
    """
    Singleton decorator

    Args:
        cls: singleton class

    Returns:
        - an instance of a singleton class
    """
    instances = {}

    @wraps(cls)
    def getinstance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return getinstance


@singleton
class Environment:
    """
    Environment is a running environment class.

    Args:
        profiling_window: profiling window size
        configs: configs for running tasks
        debug: running with debug information
        no_warning: do not output warning informations
        seed: initial seed for random and torch
        device: running device
        fp16: running with fp16
        no_progress_bar: do not show progress bar
        pb_interval: show progress bar with an interval
    """

    def __init__(self,
                 configs=None,
                 profiling_window: int = 0,
                 debug: bool = False,
                 no_warning: bool = False,
                 seed: int = 0,
                 device: str = None,
                 fp16: bool = False,
                 no_progress_bar: bool = False,
                 pb_interval: int = 1,
                 distributed: str = 'ddp',
                 backend: str = 'nccl',
                 custom_libs: str = None,
                 local_rank: int = 0,
                 log_filename: str = None):
        self.profiling_window = profiling_window
        self.configs = configs
        self.debug = debug
        self.no_warning = no_warning
        self.seed = seed
        self.fp16 = fp16
        self.no_progress_bar = no_progress_bar
        self.pb_interval = pb_interval
        self.distributed = distributed
        self.backend = backend
        self.log_filename = log_filename

        self.distributed_world = 1
        self.rank = 0
        self.local_rank = local_rank
        if device is None:
            self.device = 'cpu'
        else:
            self.device = device
        if self.device == 'cuda':
            self._init_cuda()

        self._init_log()
        self._init_seed()
        self._import_custom_lib(custom_libs)

    def _init_log(self):
        FORMAT = f'%(asctime)s ｜ %(levelname)s | %(name)s |{f" RANK {self.rank} | " if not self.is_master() else " "}%(message)s'
        level = logging.INFO if self.is_master() else logging.WARN
        logging.basicConfig(filename=self.log_filename, format=FORMAT, datefmt='%Y-%m-%d,%H:%M:%S', level=level)

    def _import_custom_lib(self, path):
        """
        Import library manually

        Args:
            path: external libraries split with `,`
        """
        if path:
            path = path.strip('\n')
            for line in path.split(','):
                logger.info(f'import module from {line}')
                line = line.replace('/', '.')
                importlib.import_module(line)

    def _init_cuda(self):
        """
        Initialize cuda device

        We assume that the user will not run ParaGen on more than one workers with only 1 GPU
        used on each worker.
        """
        if torch.cuda.device_count() > 1:
            if self.distributed in ['horovod', 'hvd']:
                import horovod.torch as hvd
                hvd.init()
                self.rank = hvd.rank()
                self.local_rank = hvd.local_rank()
                self.distributed_world = hvd.size()
            elif self.distributed == 'ddp':
                import torch.distributed as dist
                dist.init_process_group(backend=self.backend)
                self.rank = dist.get_rank()
                self.distributed_world = dist.get_world_size()
            else:
                raise NotImplementedError
        torch.cuda.set_device(self.local_rank)
        torch.cuda.empty_cache()

    def _init_seed(self):
        """
        Initialize global seed
        """
        import random
        random.seed(self.seed)
        import torch
        torch.manual_seed(self.seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed(self.seed)

    def is_master(self):
        """
        check the current process is the master process
        """
        return self.rank == 0

    def join(self):
        if self.distributed in ['horovod', 'hvd']:
            import horovod as hvd
            hvd.join()
        else:
            pass


def build_env(*args, **kwargs):
    """
    Build environment
    """
    env = Environment(*args, **kwargs)
    logger.info('Create environment with \n{}\n'.format(json.dumps({
        'device': env.device,
        'fp16': env.fp16,
        'profiling_window': env.profiling_window,
        'debug': env.debug,
        'distributed_world': env.distributed_world,
        'rank': env.rank,
        'local_rank': env.local_rank,
        'no_progress_bar': env.no_progress_bar,
        'no_warning': env.no_warning,
        "pb_interval": env.pb_interval
    }, indent=4, sort_keys=True)))


def format_states(states):
    """
    Format logging states to prettify logging information

    Args:
        states: logging states

    Returns:
        - formated logging states
    """
    formated_states = {}
    for key, val in states.items():
        if isinstance(val, float):
            if val < 1e-3:
                val = '{:.4e}'.format(val)
            else:
                val = '{:.4f}'.format(val)
        formated_states[key] = val
    return formated_states


def str_pipes(states):
    """
    Make state dict into a string

    Args:
        states: state dict

    Returns:
        - state dict in string
    """
    return " | ".join('{} {}'.format(key, states[key]).strip() for key in states.keys())


def progress_bar(iterable, streaming=False, **kwargs):
    """
    Create progress bar for iterable object

    Args:
        iterable: iterable object
        streaming: iterable object does not have __len__ property

    Returns:
        - progress bar
    """
    env = Environment()
    if env.is_master() and not env.no_progress_bar:
        total = 0 if streaming else len(iterable)
        pb = tqdm(iterable, total=total, leave=False, mininterval=env.pb_interval, **kwargs) if total > 0 else \
            tqdm(iterable, leave=False, mininterval=env.pb_interval, **kwargs)
    else:
        pb = iterable
    return pb