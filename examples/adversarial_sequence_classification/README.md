# Adversarial Sequence Classification

The example is an implementation of:
```
Chen Zhu, Yu Cheng, Zhe Gan, Siqi Sun, Tom Goldstein, and Jingjing Liu. FreeLB: Enhanced Adversarial Training for Language Understanding. In ICLR, 2020.
```

In this example, we will show:

- how to write your own code as an extension to ParaGen by making `minimal` changes to `model` and `trainer`.

- how to fully utilize all GPUs to conduct grid search over some hyper params `without` any modification to your code.

<!-- # GLUE Benchmark (on Dev Set)

Fine-tune GLUE with roberta-base

| Task | Normal Training | Adversarial Training |
|---| ---| --- |
| CoLA (Matthews' Corr) | 62.82 | |
| SST-2 (Accuracy) | 94.95 | |
| STS-B (Pearson/Spearman Corr) | 91.01/90.77 | |
| MRPC (F1/Accuracy) | 92.31/89.22 | |
| QQP (F1/Accuracy) | 89.02/91.76 | |
| MNLI-m/mm (Accuracy) | 87.72/87.38 | |
| QNLI (Accuracy) | 92.75 | |
| RTE (Accuracy) | 80.14 | 82.31 | -->

## Adversarially fine-tune `roberta-base` on GLUE

Download and prepare the data by running
```bash
cd examples/adversarial_sequence_classification/glue
bash glue-fetch.sh
cd ..
```

Train `roberta-base` on `RTE`: 
```bash
paragen-run --config glue/configs/RTE.yaml
```

## Grid-search

In ParaGen, most hyper-parameters are fixed and saved in a configuration file, e.g.,  `RTE.yaml`. 
The performance may be sensitive to some hyper-parameters, so we can **override** those args in command-line, such as:

```bash
paragen-run --config glue/configs/RTE.yaml --task.model.init_mag 0.03
```

If we want to perform grid-search on these parameters and fully utilize all GPUs, we can use `ManyTasks`:

Install requirements:

```bash
pip install git+https://github.com/dugu9sword/manytasks.git
```

Change some params in `search.json` and run grid searching:

```bash
manytasks run search
```

## Results

Under normal training, the accuracy on RTE is `80.14`,

| task.model.adv_lr |  task.model.init_mag | task.model.max_norm | Best Accuracy (bsz=16) | Best Accuracy (bsz=32) |
| --- | --- | --- | --- | --- | 
|0.05     |  0.1    | 0.1  | **80.14** | **80.86** | 
|0.1      |  0.1    | 0.1  | **80.87** | **80.50** | 
|0.05     |  0.2    | 0.2  | **80.14** | 79.78 | 
|0.1      |  0.2    | 0.2  | **80.87** | **81.58** | 

<!-- |0.05     |  0.3    | 0.3  | 79.06 | 76.53 | 
|0.1      |  0.3    | 0.3  | 78.70 | 79.06 | 
|0.05     |  0.6    | 0.6  | 82.31 | 77.97 | 
|0.1      |  0.6    | 0.6  | 78.33 | 78.70 |  -->
