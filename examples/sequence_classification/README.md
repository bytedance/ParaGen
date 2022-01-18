# Sequence Classification

# GLUE Benchmark (on Dev Set)

Fine-tune GLUE with roberta-base

| Task | Performance |
|---| ---|
| CoLA (Matthews' Corr) | 63.86 |
| SST-2 (Accuracy) | 95.41 |
| STS-B (Pearson/Spearman Corr) | 91.23/90.91 |
| MRPC (F1/Accuracy) | 92.45/89.71 | 
| QQP (F1/Accuracy) | 89.10/91.85 |
| MNLI-m/mm (Accuracy) | 88.07/87.61 |
| QNLI (Accuracy) | 92.84 |
| RTE (Accuracy) | 79.42 |

# Fine-tune a roberta-base on GLUE

Download and prepare the data by running
```bash
cd examples/sequence_classification/glue
bash glue-fetch.sh
```

Then train a glue task as 
```bash
for TASK in CoLA SST-2 MRPC QQP MNLI QNLI RTE STS-B; do
    paragen-run --config configs/${TASK}.yaml
done
```
