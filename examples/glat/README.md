# Glancing Transformer ([Qian et.al](https://arxiv.org/abs/2008.07905))

| Task | sacrebleu | tok bleu | compound bleu |
| --- | --- | --- | --- |
| WMT14 En-De | 24.59 | 24.82 | 25.14 |
| WMT14 En-De (avg ckpt) | 24.69 | 24.91 | 25.24 |

We report `compound bleu` for completeness but we recommend to use `sacrebleu` for future comparison.

# WMT14 En-De

Fetch data by 
```bash
wget https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/bycha/glat/kd_data.tgz
tar -xvf kd_data.tgz

```
The distilled data can also be downloaded from http://dl.fbaipublicfiles.com/nat/distill_dataset.zip .

Train GLAT with 
```bash
paragen-run --config configs/train.yaml --lib glat
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
