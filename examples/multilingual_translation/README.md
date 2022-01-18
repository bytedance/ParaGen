# Multilingual Translation

Multilingual translation here consists of two parts: pre-training an mBART from scratch and 
fine-tuning mBART on a multilingual translation dataset.

In this implementation, we use 32 Telsa V100 GPUs to train and finetune mBART.
Here we report results of XX-en pairs.

| Test language pairs   | [Tang et.al](https://arxiv.org/abs/2008.00401) | ParaGen  | ParaGen checkpoints average |
| ---                   | ---                       | ---    | ---                       |
| de-en                 | 41.5                      | 41.45  | 41.84                     |
| fr-en                 | 39.8                      | 39.09  | 39.41                     |
| ja-en                 | 20.5                      | 21.68  | 22.84                     |
| pl-en                 | 32.9                      | 32.03  | 32.50                     |
| ro-en                 | 38.6                      | 36.75  | 37.24                     |
| mn-en                 | 13.6                      | 12.75  | 14.14                     |
| hi-en                 | 27.2                      | 26.34  | 27.78                     |

## Pre-training mFBART from Scratch

We train mBART on [CC-100](http://data.statmt.org/cc-100/) by
```bash
paragen-run --lib mlt --config configs/pretrain_mbart.yaml
```

## Fine-tuning mBART to Multilingual Translation Data

### STEP 1 - Download Datasets

We will first download the raw datasets from  and use `preprocess_data.py` to preprocess datasets.

### STEP 2 - Finetuning mBART
In this implementation, we use 32 Telsa V100 GPUs to finetune mBART on the required language pairs.

We will train the model by:
```bash
paragen-run --lib mlt --config configs/finetune.yaml
```

### STEP 3 - Evaluation
```bash
paragen-run --lib mlt --config configs/eval.yaml
```

