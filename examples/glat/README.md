# Glancing Transformer ([Qian et.al](https://arxiv.org/abs/2008.07905))

| Task                   | sacrebleu | tok bleu |
|------------------------|-----------|----------|
| WMT14 En-De            | 24.40     | 25.01    |
| WMT14 En-De (avg ckpt) | 24.58     | 25.19    |

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
Remind keeping max tokens within a training batch to 64k to obtain comparable results.

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
