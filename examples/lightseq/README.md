# LightSeq

[LightSeq](https://github.com/bytedance/lightseq) is a high performance training and inference library for transformer.
ParaGen implements an easy way to use LightSeq to speed up training and inference for transformer model.
Here we use IWSLT14 De-En as a examples.

# LightSeq training

Above all, run data preparation process following [IWSLT14 De-En guidance](../translation).
Then copy data and vocabulary to this directory.
Train a LightSeq-optimized transformer on the data
```bash
paragen-run --config configs/train.yaml --lib ls
``` 

# LightSeq serialization
Export transformer to its LightSeq inference version by
```bash
paragen-export --config config/export.yaml --lib ls
```

After serialize your transformer, you may test its performance by
```bash
paragen-run --config config/eval.yaml --lib ls
```
