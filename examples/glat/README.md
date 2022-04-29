# Glancing Transformer ([Qian et.al](https://arxiv.org/abs/2008.07905))

| Task                   | sacrebleu | tok bleu |
|------------------------|-----------|----------|
| WMT14 En-De            | 24.40     | 24.98    |
| WMT14 En-De (avg ckpt) | 24.76     | 25.40    |
| WMT14 En-De (avg ckpt + NPD=7) | 25.29 | 25.95    |

Note: We use self-reranking instead AT-reranking for NPD=7 setting.

# WMT14 En-De

Fetch data by 
```bash
wget http://dl.fbaipublicfiles.com/nat/distill_dataset.zip
unzip distill_dataset.zip
```

Train GLAT with 
```bash
paragen-run --config configs/train.yaml --lib glat
```
Note we train our model on 8 V100-32G GPUs.
Make sure to train GLAT with `64k` tokens within a batch if you are using less GPUs.

After training, the model is evaluate with
```bash
python scripts/average_checkpoints.py --dir checkpoints --prefix last
paragen-run --config configs/eval-npd.yaml --lib glat
```

# Serialize GLAT

Fill well-trained GLAT model, vocabularies and data to configuration at `configs/export.yaml`

Export GLAT with 
```bash
paragen-export --config configs/export.yaml --lib glat
```

Finally, test traced with 
```bash
paragen-run --config configs/eval-traced.yaml --lib glat
```
