# Summarization

We benchmark summarization task on Multi-News dataset with pretrained BART-base from Hugginface.

| Task | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- |
| Multi-News | 46.80 | 17.93 | 43.01 |

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
paragen-run --config configs/train-bart-base.yaml --task.model.path {MODEL_PATH}
```

#### Evaluate model
After obtained our model, we evaluate the model by
```bash
paragen-run --config configs/eval-bart-base.yaml --lib summ
```
