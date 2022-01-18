In ParaGen, data processing functions are fully defined within the `Task` class, including `_data_collate_fn`,
`_collate` and `_output_collate_fn`.

# Data collate function

As is shown in [02_data_loading.md](./02_data_loading.md), data collate function `_data_collate_fn` is passed to `Dataset` class.
It is called as soon as each sample is fetch from a file system.
```python
from typing import Dict

from paragen.datasets import create_dataset


class TranslationTask:

    def _build_datasets(self):
        """
        Build a datasets
        """
        self._datasets = {}
        for key, configs in self._data_configs.items():
            dataset = create_dataset(configs)
            dataset.build(collate_fn=lambda x: self._data_collate_fn(x))
    

    def _data_collate_fn(self, sample: Dict) -> Dict:
        processed_sample = {}
        for key in [self._src, self._tgt]: # src and tgt refer to the translation pair
            if key in sample:
                # tokenizer encodes textual tokens into their indices
                processed_sample[key] = self._tokenizer[key].encode(sample[key])
        return processed_sample
```
The most important point of `_data_collate_fn` is that it could pre-processed to save time in training neural model.

# Batch collate function

After processing data, data are grouped into batches for further training.
The batch processing function `_collate` transforms a list of samples into a dict/list of `torch.Tensor`, and these
tensors are fed into neural model.
In usage, `_collate` function is passed into `DataLoader` for batch processing after fetching a list of samples, where
the samples are processed by `_data_collate_fn`.
```python
from paragen.dataloaders import build_dataloader
from paragen.utils.data import reorganize, split_tgt_sequence
from paragen.utils.tensor import convert_idx_to_tensor


class TranslationTask:

    def _build_dataloader(self, name):
        configs = self._dataloader_configs[name]
        sampler_configs = configs.pop('sampler') # do not build sampler if StreamingDataLoader is used.
        sampler = self._build_sampler(self._datasets[name], sampler_configs,)
        dataloader = build_dataloader(configs, dataset=self._datasets[name], sampler=sampler, collate_fn=self._collate,)
        return dataloader

    def _collate(self, samples):
        samples = reorganize(samples)
        src = samples[self._src]
        src = convert_idx_to_tensor(src, pad=self._tokenizer[self._src].pad, ndim=2)
        tgt, prev_tokens = split_tgt_sequence(samples[self._tgt], bos=self._tokenizer[self._tgt].bos, eos=self._tokenizer[self._tgt].eos)
        tgt = convert_idx_to_tensor(tgt, pad=self._tokenizer[self._tgt].pad, ndim=2)
        prev_tokens = convert_idx_to_tensor(prev_tokens, pad=self._tokenizer[self._tgt].pad, ndim=2)
        batch = {
            'net_input': {'src': src, 'tgt': prev_tokens,},
            'net_output': {'target': tgt}}
        return batch
```

For easy understanding of data pre-processing, this data processing is almost equivalent to the following simplification:
```python
data, bsz = [], 32 # data is iterable on samples
for i, line in enumerate(data): 
    data[i] = data_collate_fn(line)
batches = [data[i*bsz:(i+1)*bsz] for i in range(len(data) // bsz)]
for i, batch in enumerate(batches):
    batches[i] = collate_fn(batch)
```

# Output collate function

After a batch of samples in tensor is processed by the neural model, the `Task` class also includes process on neural 
model output in tensor.
```python
from paragen.utils.tensor import convert_tensor_to_idx


class TranslationTask:

    def _output_collate_fn(self, outputs, *args, **kwargs):
        outputs = convert_tensor_to_idx(outputs,
                                        bos=self._tokenizer[self._tgt].bos,
                                        eos=self._tokenizer[self._tgt].eos,
                                        pad=self._tokenizer[self._tgt].pad)
        processed_outputs = []
        for output in outputs:
            output = self._tokenizer[self._tgt].decode(output)
            processed_outputs.append(output)
        return processed_outputs
```

The `_output_collate_fn` is called by `Evaluator` class after neural model output results.
For examples, a translation model (`SequenceGenerator`) produces a translation sequence given source sentence.
It first converts the tensors into index, and then transforms index back into their textual format.

# Tokenizer

`Tokenizer` class is a special class used in data processing.
It is used to transform a textual input into an indexed one (`encode`) and vice versa (`decode`).
For a `SentencePieceTokenizer` example, it encodes and decodes a sequence via `sentencepiece` lib.
```python
from typing import List

from paragen.tokenizers import AbstractTokenizer, register_tokenizer

@register_tokenizer
class SentencePieceTokenizer(AbstractTokenizer):

    def __init__(self,
                 spm_path=None,):
        import sentencepiece
        self._sp = sentencepiece.SentencePieceProcessor()
        self._sp.Load(spm_path)

    def encode(self, x, *args) -> List[int]:
        return self._sp.Encode(x)
        

    def decode(self, x: List[int]) -> str:
        return self._sp.Decode(x)
```
Besides, it also contains a `staticmethod` function `learn` for learning the tokenizer from a certain dataset.
