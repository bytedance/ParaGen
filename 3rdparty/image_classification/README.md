# Image Classification

## Cifar-100

| Task     | top-1 Acc (%) |
|----------|---------------|
| resnet50 | 60.8          |

Download CIFAR-100 from https://www.cs.toronto.edu/~kriz/cifar.html.

Process data with `python scripts/cifar_data.py`

Then train the `resnet50` with `paragen-run --config configs/train.yaml --lib image`

