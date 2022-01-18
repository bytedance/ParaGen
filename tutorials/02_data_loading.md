ParaGen uses three main classes for data loading: `Dataset`, `Sampler` and `DataLoader`.
An overview of ParaGen data loading module can be concluded as follows.
For example, given a pair of large-scale parallel file `text.en` and `text.zh`, a `ParallelDataset` can be created with 
```python
from paragen.dataloaders.streaming_dataloader import StreamingDataLoader
from paragen.datasets.streaming_text_dataset import StreamingTextDataset

dataset = StreamingTextDataset(path='text')
dataset.build()
dataloader = StreamingDataLoader(dataset, 
                                 collate_fn=None,  # collate_fn is a processing function to data produced by `dataset`
                                 batch_size=8,
                                 )

for batch in dataloader:
    pass  # do something to current batch of samples. You have sample in json-format, such as {'en': 'hello', 'zh': '你好'}
```

# Dataset

Dataset is designed to read data from local disk and hadoop distributed file system.
Datasets in ParaGen differ in the data format.
For example, `ParallelDataset` is used to read parallel data stored in different files where each part of data
is aligned by their line number. 
A typical case of `ParallelDataset` is bilingual machine translation, which take two files of translation pairs.

Given a pair of parallel file `text.en` and `text.zh`, a `ParallelDataset` can be created with
```python
from paragen.datasets.parallel_text_dataset import ParallelTextDataset

dataset = ParallelTextDataset(path={'en': 'text.en', 'zh': 'text.zh'}) # create data io class
dataset.build() # build dataset by passing data processing options
```

## How different classes of datasets become different

The difference in different classes of datasets lies in the `_callback` function within their classes.
Let's see the `_callback` functions of `ParallelTextDataset` and `JsonDataset`. 
```python
class ParallelTextDataset:

    def _callback(self, sample):
        sample = {key: [eval(v) for v in val] for key, val in sample.items()}
        return sample

import json

class JsonDataset:

    def _callback(self, sample):
        sample = json.loads(sample)
        return sample

```
Note `_callback` function is fully decided by data format and do nothing with a specific machine learning task.

## Pass task-specific processing function

The dataset also accepts a task-specific function as an arguments to further process data. 
The passage of functions enables task-specific data processing before model training, enabling fast running 
in downstream training.
For example, we pass a function add a `!` suffix to each value.
```python
dataset.build(lambda x: {k: f'{v}!' for k, v in sample.items()})

for sample in dataset:
    pass # do something you want. You have processed sample, such as {'en': 'hello!', 'zh': '你好!'}
```

When a task instance builds up a data set, it usually passes its `_data_collate_fn` function to its dataset.
```python
from typing import Dict

class TranslationTask:

    def _data_collate_fn(self, sample: Dict) -> Dict:
        processed_sample = {}
        for key in [self._src, self._tgt]: # src and tgt refer to the translation pair
            if key in sample:
                # tokenizer encodes textual tokens into their indices
                processed_sample[key] = self._tokenizer[key].encode(sample[key])
        return processed_sample
```

# Sampler

Samplers produce the sampling order with respect to a given dataset.
One critical property in Sampler is `permutation`, which stores the index (typed as `int`) of samples, 
and samples are enumerated according to the `permutation` list.
```python
from torch.utils.data import Sampler

class AbstractSampler(Sampler):
    """
    Sampler produces the sample indexes for sequential training.
    Any strategy can be used to produces a permutation list to define the order to fetch samples.

    """
    def __getitem__(self, idx):
        return self._permutation[idx]
```
Mostly, samplers are usually used via `batch_sampler` interface, which returns a batch list typed as `List[List[int]]`.
Note each batch is represented as a list of indices.
Usually, `batch_sampler` can inherit from `torch.utils.data.Sampler.batch_sampler` by grouping samples 
automatically.
It can also be customized by overwriting this method. 

# DataLoader

DataLoaders takes a dataset and a sampler to produce batches iteratively.
Different from `torch.utils.data.DataLoader`, a ParaGen-style dataloader requires a collate function `collate_fn`
to further process a batch of samples to make it consumable for neural network.
```python
from torch.utils.data import DataLoader as D

class DataLoader(D):
    
    def __iter__(self):
        for samples in super().__iter__():
            yield self._collate_fn(samples)
```

# Streaming and In-Memory Data Loading

One important issue in data loading is large-scale data.
When the data size becomes large, storing all these data into memory may raise OOM problem.
Thus we design another style of data loading mechanism, namely streaming data processing.

## StreamingDataset

Different from `InMemoryDataset`, `StreamingDataset` keeps only an input stream.
`StreamingDataset` reads a piece of data only when `DataLoader` is creating a new batch of neural inputs during training,
while `InMemoryDataset` reads all the data before starting training.

`InMemoryDataset` contains a `_load` function called in data set creation.
```python
from torch.utils.data import Dataset as D

class InMemoryDataset(D):

    def __iter__(self):
        for sample in self._data:
            yield sample

class JsonDataset(InMemoryDataset):

    def _load(self):
        """
        Preload all the data into memory. In the loading process, data are preprocess and sorted.
        """
        fin = open(self._path)
        self._data = []
        for i, sample in enumerate(fin):
            sample = sample.strip('\n')
            self._data.append(self._callback(sample))
        fin.close()
```
But `StreamingDataset` outputs data by fetching from input stream instantly.
```python
from torch.utils.data import Dataset as D

class StreamingDataset(D):

    def __iter__(self):
        for sample in self._fin:
            yield self._callback(sample)
```

## StreamingDataLoader

Above all, let's see a configuration example of `InMemoryDataLoader` and `SteramingDataLoader`.
```yaml
# InMemoryDataLoader configs
task:  
  dataloader:
    train:
      class: InMemoryDataLoader
      sampler:
        class: ShuffleSampler
        max_tokens: 16000
# StreamingDataLoader configs
task:  
  dataloader:
    train:
      class: StreamingDataLoader
      max_tokens: 16000
      length_interval: 8 
      maxlen: 256
      max_shuffle_size: 64
```

It is observed that unlike `InMemoryDataLoader`, `StreamingDataLoader` does not require a sampler over a given dataset.
Because, the size of a streaming data set is unexpected, and corpus-level sampling strategies are hard 
to apply to the dataset.
Thus we incorporate a simple local-shuffling strategy into a StreamingDataLoader.
The arguments `length_interval`, `maxlen` and `max_shuffle_size` are the ones for local-shuffling strategy.

### Post Collate
A critical issue in using `StreamingDataLoader` is when to apply `collate_fn` to a batch.
In original `torch.data.utils.DataLoader`, it only supports batch-by-samples strategies or `batch_sampler` option,
both of which fail to use batch-by-token strategy on large-scale data.
But batch-by-token strategy significantly enhances training process 


# Customizing Dataset, Sampler and DataLoader

The general customization tutorial could be found at [01_arguments_specification.md](./01_arguments_specification.md)
To create you own dataset class, you can follow the examples below.
For simplicity, we demonstrate with a `InMemoryDataset`.
```python
from paragen.datasets import register_dataset
from paragen.datasets.in_memory_dataset import InMemoryDataset

@register_dataset
def MyDataset(InMemoryDataset):

    def __init__(self, *arg, **kwargs): # define your own arguments
        pass

    def build(self, collate_fn, *args, **kwargs): # define which resources are used to process data
        pass
        
    def _callback(self): # parse sample with your own format
        pass
        
    def __iter__(self): # produce samples iteratively.
        pass

```

To customize a sampler, the most important thing is the computation of `permutation` attributes 
and `batch_sampler` property function.
```python
from paragen.samplers import AbstractSampler, register_sampler


@register_sampler
class MySampler(AbstractSampler):

    def __init__(self, *args, **kwargs): # define your own arguments
        pass

    def build(self, data_source): # create permuation attribute here
        pass

    def reset(self, *args, **kwargs): # change sampling strategy according to outside states
        pass
```

For dataloader, we do not recommend users to write it by self.
But it could be useful to rewrite the data loader.
One important function of a dataloader is to create batches of processed samples.
Thus two things should be done by dataloader:
- determine which samples are grouped into a batch;
- determine which batch should be returned outside;
- apply data processing function to a batch of samples.

These objectives are all implemented in a `__iter__` function.
```python
from paragen.dataloaders import AbstractDataLoader, register_dataloader

@register_dataloader
def MyDataLoader(AbstractDataLoader): # inheriting from InMemoryDataLoader or StreamingDataLoader is also a good choice

    def __init__(self, *args, **kwargs): # define your own arguments
        pass
    
    def __iter__(self): # write your own iterative algorithm
        iterator = super().__iter__()
        pass
```
