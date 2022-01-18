from paragen.dataloaders import build_dataloader
from paragen.datasets import create_dataset
from paragen.evaluators import create_evaluator
from paragen.generators import create_generator
from paragen.samplers import create_sampler
from paragen.tasks import AbstractTask
from paragen.tokenizers import create_tokenizer
from paragen.trainers import create_trainer
from paragen.utils.ops import deepcopy_on_ref


class BaseTask(AbstractTask):
    """
    BaseTask defines overall scope on general tasks, namely, fitting a model to a dataset.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_tokenizers(self):
        """
        Build tokenizers
        """
        self._tokenizer = create_tokenizer(self._tokenizer_configs)
        self._tokenizer.build()

    def _build_datasets(self):
        """
        Build a datasets
        """
        self._datasets = {}
        for key, configs in self._data_configs.items():
            dataset = create_dataset(configs)
            if key == 'train':
                dataset.build(collate_fn=lambda x: self._data_collate_fn(x, is_training=True),
                              preprocessed=self._preprocessed)
            else:
                dataset.build(collate_fn=lambda x: self._data_collate_fn(x, is_training=False),
                              preprocessed=self._preprocessed)
            self._datasets[key] = dataset

    def _build_sampler(self, dataset, configs, is_training):
        """
        Build a data sampler

        Args:
            dataset: dataset instance
            configs: sampler configuration
        """
        sampler = create_sampler(configs, is_training=is_training)
        sampler.build(dataset)
        return sampler

    def _build_generator(self):
        """
        Build generator for model in inference
        """
        self._generator = create_generator(self._generator_configs)
        self._generator.build(self._model)

    def _build_dataloader(self, name, mode):
        """
        Build dataloader

        Args:
            name: data name
            mode: running mode for data loader

        Returns:
            - dataloader instance

        """
        configs = deepcopy_on_ref(self._dataloader_configs[name])
        if 'sampler' in configs:
            sampler_configs = configs.pop('sampler')
            sampler = self._build_sampler(self._datasets[name],
                                          sampler_configs,
                                          mode == 'train')
        else:
            sampler = None
        dataloader = build_dataloader(configs,
                                      dataset=self._datasets[name] if name in self._datasets else None,
                                      sampler=sampler,
                                      collate_fn=self._collate,
                                      post_collate=self._post_collate)
        return dataloader

    def _build_trainer(self):
        """
        Build a trainer to schedule training process
        """
        dataloader = self._build_dataloader('train', mode='train')
        eval_dataloaders = {}
        for name, configs in self._dataloader_configs.items():
            if name != 'train':
                eval_dataloaders[name] = self._build_dataloader(name, mode='valid')
        self._trainer = create_trainer(self._trainer_configs)
        self._trainer.build(model=self._model,
                            dataloader=dataloader,
                            criterion=self._criterion,
                            eval_dataloaders=eval_dataloaders,
                            evaluator=self._evaluator,
                            task_callback=self._callback)

    def _build_evaluator(self):
        """
        Build a evaluator to schedule evaluation process
        """
        dataloaders = {}
        for name, configs in self._dataloader_configs.items():
            if name != 'train':
                dataloaders[name] = self._build_dataloader(name, mode='infer')
        self._evaluator = create_evaluator(self._evaluator_configs)
        self._evaluator.build(generator=self._generator,
                              dataloaders=dataloaders,
                              tokenizer=self._tokenizer,
                              task_callback=self._callback,
                              postprocess=self.postprocess)

    def _callback(self, training, infering=False):
        """
        Context management callback

        Args:
            training: is training
            infering: is infering
        """
        self._training = training
        self._infering = not training and infering
