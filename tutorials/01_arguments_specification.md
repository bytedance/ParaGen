In ParaGen, arguments are passed to a program via a `yaml` file.

# Arguments Definition
The first question to use ParaGen is how to define arguments and how to pass arguments.
For example, we create a translation task named `TranslationTask`
```python
from paragen.tasks import register_task

@register_task
class TranslationTask:
    """
    Seq2SeqTask defines overall scope on sequence to sequence task.

    Args:
        src: source language
        tgt: target language
        maxlen: max length for input sequences
    """

    def __init__(self,
                 src,
                 tgt,
                 maxlen=512
                 ):
        pass
```
As is show in examples, `TranslationTask` requires three arguments (`src`, `tgt`, `maxlen`) to initialize
its instance.
Then we may pass the arguments by
```yaml
task:  
    class: TranslationTask
    maxlen: 256
    src: en
    tgt: zh
```
Note there is a special key `class` specifying which task to initialize.

Besides, the `yaml` configurations can be recursively refined.
For example, we create a seq2seq model named `Seq2Seq`
```python
from paragen.models import register_model


@register_model
class Seq2Seq:
    """
    EncoderDecoderModel defines overall encoder-decoder architecture.

    Args:
        encoder: encoder configurations to build an encoder
        decoder: decoder configurations to build an decoder
        d_model: feature embedding
        share_embedding: how the embedding is share [all, decoder-input-output, None].
            `all` indicates that source embedding, target embedding and target
             output projection are the same.
            `decoder-input-output` indicates that only target embedding and target
             output projection are the same.
            `None` indicates that none of them are the same.
        path: path to restore model
    """

    def __init__(self,
                 encoder,
                 decoder,
                 d_model,
                 share_embedding=None,
                 path=None):
        super().__init__(encoder=encoder,
                         decoder=decoder,
                         d_model=d_model,
                         share_embedding=share_embedding,
                         path=path)
```
If we want to pass the neural model configuration to our translation task, we first add model configs
to `TranslationTask` by
```python
class TranslationTask:

    def __init__(self,
                 src,
                 tgt,
                 model,
                 maxlen=512
                 ):
        pass
```
Then we set `yaml` file to
```yaml
task:  
    class: TranslationTask
    src: en
    tgt: zh
    maxlen: 256
    model:
      class: Seq2Seq
      encoder: ...
      decoder: ...
      d_model: 512
      share_embedding: decoder-input-output
```

# Overwrite Default Arguments

ParaGen supports to high-priority arguments to overwrite default ones.
For example, if we want to run a `de-fr` task with a 256d neural model, run paragen with
```bash
{PARAGEN_COMMAND} --config {CONFIG_PATH} --task.src de --task.tgt fr --task.model.d_model 256
```

# Create a new class
In ParaGen, there are mainly 12 special categories of classes, including
- `criterion` computes training loss and is used for optimization;
- `dataloader` inherits from `torch.utils.data.DataLoader` and is used to create batches of samples;
- `dataset` inherits from `torch.utils.data.Dataset` and is to read data from `path`;
- `evaluator` schedules a evaluation process;
- `generator` combines a neural model with inference algorithms and enable an end-to-end inference;
- `metric` assesses neural model (`generator`) output by comparing with ground truth;
- `model` is learned from data;
- `optimizer` is used to learn neural model given a criterion;
- `sampler` is built over a dataset and yields samples with a sampling strategy;
- `task` defines every components of a running process;
- `tokenizer` transform data from its raw format to a tensor one;
- `trainer` schedules a training process.

For each special category, there is a `register_{class}` decorator to add the target class into ParaGen module list.
Besides, each category has a `Abstract{Class}` defining its interfaces, and a specific task should inherit from it.
For example, we are creating a `MyTask`.
```python
from paragen.tasks import register_task, AbstractTask

@register_task
class MyTask(AbstractTask):

    pass # implements all interfaces required

```
Then you are able to refer your own task by
```yaml
task:
    class: MyTask # the class name is case-insensitive
```

# Environment Arguments

We also set an environment domain for keeping globally accessible variables, such as overall configuration,
distributed world size and usage of fp16.
We do not recommends to put task-specific variables to `env`, but it will be useful in some scenarios.
