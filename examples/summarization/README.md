# Summarization

We benchmark summarization task on Multi-News and XSum datasets.

| Task | Model | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | -- | --- | --- |
| Multi-News | Transformer-base (greedy search) | 33.59 | 5.91 | 30.71 |
| Multi-News | bart-base w/o pretrain | 38.43 | 8.53 | 35.02 |
| Multi-News | bart-base | 46.80 | 17.93 | 43.01 |
| XSum | bart-base | 42.49 | 19.52 | 34.37 |

## Dependency

To reproduce the summarization results, `pyrouge` is required:
```bash
pip install pyrouge
```

### Multi-News

Download and process the Multi-News dataset from [Huggingface](https://huggingface.co/datasets/multi_news).

### Preprocess Dataset

We first preprocess dataset for fast data loading in training:
```bash
paragen-preprocess --config configs/preprocess.yaml
```

#### Train BART-base model
Then we train bart-base model on preprocessed data
```bash
paragen-run --config configs/train-bart-base.yaml
```

#### Evaluate model
After obtained our model, we evaluate the model by
```bash
paragen-run --config configs/eval-bart-base.yaml --lib summ --task.model.path {MODEL_PATH}
```
