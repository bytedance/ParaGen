# Question Answering

## SQuAD 1.1

Fine-tune on SQuAD 1.1. 

| Model | F1 | Exact Match |
|---|---|---|
| bert-base-uncased | 88.31 | 81.02 |
| bert-base-cased | 88.34 | 81.11 |
| bert-large-cased | 90.73 | 83.87 |
| roberta-base| 91.88 | 85.34 |
| roberta-large | 93.44 | 87.26 | 
| bart-base | 91.27 | 84.52 |
| bart-large |  93.14 | 86.70 |


### Fine-tune Huggingface Question Answering Modle on SQuAD 2.0

Download and prepare the data by running
```bash
cd examples/question_answering/squad1.1/
bash prepare_squad1.1.sh
```

Then train with 
```bash
paragen-run --config train.yaml
```

### Use other pretrained model

Change `tokenizer_name` in tokenizer configuration and `pretrained_model` in model configuration.
For `base` model, we set `max_samples` to `8` and `total_steps` to `15k`; 
for `large` model, we set `max_samples` to `4` and `total_steps` to `44k`.

Now only `bert-based-uncased` is well-tuned.
Other pretrained models use the same setting as `bert-based-uncased` and requires further tuning.
