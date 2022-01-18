from contextlib import contextmanager
from typing import Dict, List
import logging
logger = logging.getLogger(__name__)

from paragen.tasks import TRAIN, EVALUATE


class AbstractTask:
    """
    Task defines overall scope on all the modules to train/evaluate/serve a task.

    Args:
        mode (str): process mode. Options: [train, valid, evaluate, serve]
        model (dict): model configuration to build a neural model
        data (dict): data configuration to build a dataset for train, valid, evaluate, serve
        dataloader (dict): dataloader configuration to build a dataloader to fetch data from dataset
            with sampling strategies
        tokenizer (dict): tokenization configuration to build a tokenizer to preprocess
        criterion (dict): criterion configuration to build a criterion to compute objective function for a model
        generator (dict): generator configuration to build a generator to produce results in inference
        trainer (dict): trainer configuration to build a trainer to train a model with criterion and optimizer
        evaluator (dict): evaluator configuration to build a evaluator to evaluate the performance of the model
        preprocessed (bool): the data set has been processed
        post_collate (bool): do collate_fn after sampling
    """

    def __init__(self,
                 mode: str,
                 model: Dict = None,
                 data: Dict = None,
                 dataloader: Dict = None,
                 tokenizer: Dict = None,
                 criterion: Dict = None,
                 generator: Dict = None,
                 trainer: Dict = None,
                 evaluator: Dict = None,
                 preprocessed: bool = False,
                 post_collate: bool = False,
                 ):
        self._mode = mode.upper()
        self._model_configs = model
        self._data_configs = data
        self._dataloader_configs = dataloader
        self._tokenizer_configs = tokenizer
        self._criterion_configs = criterion
        self._generator_configs = generator
        self._trainer_configs = trainer
        self._evaluator_configs = evaluator
        self._preprocessed = preprocessed
        self._post_collate = post_collate

        self._tokenizer = None
        self._datasets = None
        self._model = None
        self._criterion = None
        self._generator = None
        self._trainer, self._evaluator = None, None
        self._training, self._infering = False, False

    def build(self):
        """
        Build necessary modules if their configs are provided.
        """
        if self._tokenizer_configs is not None:
            self._build_tokenizers()
        if self._mode in [TRAIN, EVALUATE] and self._data_configs is not None:
            self._build_datasets()
        if self._model_configs is not None:
            self._build_models()
        if self._mode in [TRAIN] and self._criterion_configs is not None:
            self._build_criterions()
        if self._generator_configs is not None:
            self._build_generator()
        if self.mode in [TRAIN, EVALUATE] and self._evaluator_configs is not None:
            self._build_evaluator()
        if self._mode in [TRAIN] and self._trainer_configs is not None:
            self._build_trainer()

    def run(self):
        """
        Run the task.
        """
        logger.info('Starting Running {} in {} mode'.format(self.__class__.__name__, self._mode))
        if self._mode == TRAIN:
            self._train()
        elif self._mode == EVALUATE:
            self._evaluate()
        else:
            logger.error('mode "{}" is not supported in run'.format(self._mode))

    def _build_tokenizers(self):
        """
        Build tokenizers
        """
        raise NotImplementedError

    def _build_datasets(self):
        """
        Build a datasets
        """
        raise NotImplementedError

    def _build_models(self):
        """
        Build one or more models
        """
        raise NotImplementedError

    def _build_sampler(self, dataset, configs, is_training):
        """
        Build a data sampler
        """
        raise NotImplementedError

    def _build_criterions(self):
        """
        Build one or more criterions
        """
        raise NotImplementedError

    def _build_generator(self):
        """
        Build generator for model in inference
        """
        pass

    def _build_trainer(self):
        """
        Build a trainer to schedule training process
        """
        raise NotImplementedError

    def _build_evaluator(self):
        """
        Build a evaluator to schedule evaluation process
        """
        raise NotImplementedError

    def _train(self):
        """
        Train neural models
        """
        self._trainer.train()

    def _evaluate(self):
        """
        Eval neural models
        """
        self._evaluator.eval()

    def preprocess(self, samples):
        """
        Serve samples with a neural model
        """
        samples = [self._data_collate_fn(sample, is_training=self._training) for sample in samples]
        samples = self._collate(samples)
        return samples

    def postprocess(self, samples, *args, **kwargs):
        samples = self._output_collate_fn(samples, *args, **kwargs)
        return samples

    def _data_collate_fn(self, sample: Dict, is_training=False) -> Dict:
        """
        Process a sample statically, such as tokenization

        Args:
            sample: a sample

        Returns:
            sample: a processed sample
        """
        return sample

    def _collate(self, samples: List[Dict]):
        raise NotImplementedError

    def _output_collate_fn(self, sample, *args, **kwargs):
        """
        Post process a sample

        Args:
            sample: an outcome

        Returns:
            sample: a processed sample
        """
        return sample

    @contextmanager
    def _context_callback(self, *args, **kwargs):
        """
        Context management callback
        """
        pass

    def export(self, path, **kwargs):
        """
        Export model for service

        Args:
            path: export path
        """
        raise NotImplementedError

    @property
    def mode(self):
        return self._mode

    def reset(self, training, infering):
        self._training = training
        self._infering = infering
