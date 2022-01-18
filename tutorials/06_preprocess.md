When training a neural model on large-scale data, the data processing would be a bottleneck of training speed.
Thus it would be useful to pre-process data before starting training by `paragen-preprocess` and `paragen-binarize-data`.

# Data Preprocess

Data preprocess can be achieved by `paragen-preprocess`.
It reads data from file(s) and process data with `task._data_collate_fn`.
```yaml
task:
  ...
dataset:
  class: JsonDataset
data:
  train:
    path: train.json
    output_path: train.index.json
  valid:
    path: valid.json
    output_path: valid.index.json
```
ParaGen supports preprocessing for multiple files by add a data configuration as one key in `data` domain.

To use preprocessed data, specify `preprocssed` argument in `yaml`.
```yaml
task:
  preprocessed: True
  ...
```

When running the task, the task pass the `preprocessed` argument into data.
```python
from paragen.datasets import create_dataset


class TranslationTask:  
    
    def __init__(self, preprocessed=False, **kwargs):
        super().__init__(**kwargs)
        self._preprocessed = preprocessed  
    
    def _build_datasets(self):
        """
        Build a datasets
        """
        self._datasets = {}
        for key, configs in self._data_configs.items():
            dataset = create_dataset(configs)
            dataset.build(collate_fn=lambda x: self._data_collate_fn(x),
                          preprocessed=self._preprocessed)
            self._datasets[key] = dataset
```

# Pre-Batch

Pre-batch can be achieved with `paragen-binarize-data`.
It processes original data until batches are all created.
To pre-batched the data, the `yaml` file can be writed as:
```yaml
task:
  ...
  dataloader:
    train:
      class: InMemoryDataLoader
      sampler:
        class: ShuffleSampler
        max_tokens: 16000
  data:
    train:
      class: ParallelTextDataset
      sort_samples: True
      path:
        de: data/train.de
        en: data/train.en
output_path: train.bin
```
Note only one data set is used to pre-compute batches.
It produces a json line for each sample, which is actually a dict of tensors.

To train neural model with pre-batched data, you need to remove the `data` domain and change the `dataloader` configs.
You may use `BinarizedDataLoader` to read pre-batched data directly, and its data are no longer required.
```yaml
task:
  ...
  dataloader:
    train:
      class: BinarizedDataLoader
      path: train.bin
      preload: True # read all data into memory
    valid:
      class: InMemoryDataLoader
      sampler:
        class: SequentialSampler
        max_samples: 128
    test:
      class: InMemoryDataLoader
      sampler:
        class: SequentialSampler
        max_samples: 128
  data:
    valid:
      class: ParallelTextDataset
      sort_samples: True
      path:
        de: data/valid.de
        en: data/valid.en
    test:
      class: ParallelTextDataset
      sort_samples: True
      path:
        de: data/test.de
        en: data/test.en
```

This is what `BinarizedDataLoader` actually does.
```python
import json
from paragen.utils.tensor import list2tensor

class BinarizedDataLoader:

    def __iter__(self):
        self._batches = open(self._path)
        for batch in self._batches:
            if not self._preload:
                batch = json.loads(batch)
                batch = list2tensor(batch)
            yield batch
```
