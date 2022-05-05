# Neural Machine Translation

Machine Translation performance with predefined configs. 
By default, the presented results are derived via model selection on validation data.

| Task | sacrebleu | tok bleu | 
| --- | --- | --- |
| IWSLT14 De-En | 33.1 | 34.5 |
| WMT14 En-De (base) | 26.9 | 27.5 |
| WMT14 En-De (big) | 27.7 | 28.4 |
| WMT14 En-Fr (big) | 38.8 | 41.7 |
| WMT14 En-Fr (big) (last10) | 40.3 | 43.3 |

To obtain `sacrebleu`, set `--task.post_detok True --task.evaluator.metric.bleu.no_tok False`.

## Training a new model

### IWSLT'14 German to English

The following instructions can be used to train a Transformer model on the [IWSLT'14 German to English dataset](http://workshop2014.iwslt.org/downloads/proceeding.pdf).

#### Download and preprocess the data
```bash
# Download and prepare the data
cd examples/translation/iwslt14.de-en
bash iwslt14.de-en.sh
```

#### Compute tokenizer resources
```bash
# compute tokenizer resources
paragen-build-tokenizer --config configs/build_tokenizer.yaml
```
Any registered tokenizer within ParaGen is allowed to build from this scripts.
The tokenizer resources can also be built from external scripts, 
such as [FastBPE](https://github.com/glample/fastBPE).

#### Train a transformer model
```bash
paragen-run --config configs/train.yaml
```
In the training process, all the checkpoints are stored.
Besides, it also records the best model.

For fast training on large scale data, you may preprocess data with 
```bash
paragen-preprocess --config configs/preprocess.yaml
```
And then run the train scripts by setting `--task.preprocessed=True` in dataset configs in `*.yaml`.

#### Evaluate model
To evaluate a model, run
```bash
paragen-run --config configs/eval.yaml
```
In translation task, we use `sacrebleu` to evaluate the results by default.
If you prefer `tokenized bleu`, set `--task.post_detok False --task.evaluator.metric.bleu.no_tok True`.

#### Export the saved model 
```bash
# serialzie and export model
paragen-export --config configs/export.yaml
```

#### Serve the serialized model
```bash
# build a server for exported model
paragen-serve-model --config configs/serve.yaml &
sleep 60
paragen-serve --config configs/serve.yaml
```

### WMT'14 English to French

The following instructions can be used to train a Transformer model on the [IWSLT'14 German to English dataset](http://workshop2014.iwslt.org/downloads/proceeding.pdf).

#### Download and preprocess the data
```bash
# Download and prepare the data
cd examples/translation/wmt14.en-fr
bash wmt14.en-fr.sh
```

#### Compute vocabulary
```bash
# compute tokenizer resources
paragen-build-tokenizer --config configs/build_tokenizer.yaml
```
Any registered tokenizer within ParaGen is allowed to build from this scripts.
The tokenizer resources can also be built from external scripts, 
such as [FastBPE](https://github.com/glample/fastBPE).

#### Train a transformer model
```bash
paragen-run --config configs/train.yaml
```
In the training process, all the checkpoints are stored.
Besides, it also records the best model.

For fast training on large scale data, you may preprocess data with 
```bash
paragen-preprocess --config configs/preprocess.yaml
```
And then run the train scripts by setting `--task.preprocessed=False` in dataset configs in `*.yaml`.

#### Evaluate model
```bash
paragen-run --config configs/eval.yaml
```

#### Export the saved model 
```bash
# serialzie and export model
paragen-export --config configs/export.yaml
```

We highly recommend to use `en-fr` configurations to train with large-scale data, 
which is robust to memory size.
