# Pretraining BERT from Scratch

## STEP 0 - Install Dependencies

We will firstly download & install `mosesdecoder`, `sentencepiece` and `datasets` for data preparation.

```
bash s0_install_deps.sh
```

## STEP 1 - Download Datasets


### WikiPedia & BookCorpus

Bert is trained on `Wikipedia` & `BookCorpus`, we can download them with `datasets`.

We filter the data with some rules:

- For the `Wikipedia` dataset, we remove some sections (`References`, `External Links`, `Categories`, etc.) and too short sentences (length<10);

- For the `BookCorpus` dataset, we remove too short sentences (length<10). (Note that this dataset may not be publicly available.)

We randomly extract 3000 documents from the `Wikipedia` dataset as the valid set. There may exist some concerns since the valid set is contained by the training set. However, (I think) it does not matter since for pretraining tasks, the valid set is not used for model selection but just serve as an indicator of the training process. Generally speaking, training on more data with larger batches for a longer time cause better results.

```bash
# download and filter the training data
python3 s1_prepare_data.py --data wiki --num_workers 40 
python3 s1_prepare_data.py --data book --num_workers 40 
cat wiki.raw-0 book.raw-0 > train.raw-0
rm wiki.raw-0 book.raw-0

# download and filter the valid data
python3 s1_prepare_data.py --data valid --num_workers 40 
```


### Newscrawl 2018-2020

You can download doc-level dataset `newscrawl` from the WMT website ( http://data.statmt.org/news-crawl/README ). There are more than 70 GB texts since 2007, the processing scripts are omitted here.



## STEP 1 - Clean and Tokenize Your Data

> **DIFFERENCE** 
>
> - Bert splits words by punctuations greedily, so `'s` will be split into `'` and `s`.
> - RoBERTa directly employs the `gpt2-bpe`, and does not do tokenization firstly.

Do punctuation normalization and tokenization with `mosesdecoder`.

```bash
# args:                  lang   input         output
bash s1_clean_and_tok.sh en     train.raw-0   train.tok-1
bash s1_clean_and_tok.sh en     valid.raw-0   valid.tok-1
```

## STEP 2 - Learn a Sentencepiece Model

Then we learn a `sentencepiece` model from the data, two files will be generated: `en.sp.model` and `en.sp.vocab`:

- `en.sp.vocab` is a text file which contains the vocabulary we will use later
- `en.sp.model` is used by the sentencepiece tokenizer to split a word into subwords

```bash
# args:                  lang   input         vocab size
bash s2_learn_spm.sh     en     train.tok-1   32000
```

## STEP 3 - Tokenize the Data with Sentencepiece

After learning the sentencepiece model, we apply it to the file.

```bash
# args:                  spm-model   input         output      num_threads
bash s3_apply_spm.sh     en.sp.model train.tok-1   train.sp-2  40
bash s3_apply_spm.sh     en.sp.model valid.tok-1   valid.sp-2  1
```

## STEP 4 - Generate Block-Like Data

The data fed into Bert look like a block, which means several sentences are combined as a long sentence (512). 
We set the `maxlen` to be `510` since two spaces are left for `<s>` and `</s>` (`[CLS]` and `[SEP]` in Bert).

```
python3 s4_gen_blocks.py --input valid.sp-2 --output valid.bl-3 --maxlen 510
```

## STEP 5 - Training BERT

In this implementation, we use 32 Telsa V100 GPUs to train two BERT models:

- `Bycha-BERT-news`: a model trained with the `newscrawl` dataset from 2018-2020.
- `Bycha-BERT-wibo`: a model trained with `wikipedia` and `bookcorpus`.

> **DIFFERENCE** 
>
> - `Bert` has "segment embeddings" while `RoBERTa`, `Bycha-BERT-news` and `Bycha-BERT-wibo` do not.
> - `RoBERTa` and `Bycha-BERT-wibo` has "embedding layernorm" while `Bert` and `Bycha-BERT-news` do not.
> - `Bert`, `RoBERTa`, `Bycha-BERT-news`, `Bycha-BERT-wibo` employ "post-layernorm".

```
paragen-run --config train.yaml
```

## STEP 7 - Finetuning BERT on Downstream Tasks

Follow `examples/sentence_classification` to prepare the GLUE datasets. For most datasets (except SST-2 and MRPC), we have to tokenize them with `mosesdecoder` firstly before applying the sentencepiece model, otherwise the performance may be hurt.

> **SOURCE**
>
> - The result of BERT is from Huggingface's website: https://huggingface.co/transformers/v1.1.0/examples.html
> - The result of RoBERTa is from `examples/sequence_classification/README.md`



| Task                          | tokenized | Official RoBERTa | Official BERT | Bycha-BERT-news  | Bycha-BERT-wibo   | cost     |
| ---                           | ---       | ---              | ---           | ---              | ---               | ---      |   
| CoLA (Matthews' Corr)         | No        | 63.86            | 57.29         | 61.82            | 58.36             | 0.5 min  |  
| SST-2 (Accuracy)              | Yes       | 95.41            | 93.00         | 93.00            | 91.86             | 3 min    |
| STS-B (Pearson/Spearman Corr) | No        | 91.23/90.91      | 89.70/89.37   | 86.97/86.88      | 87.95/87.65       | 0.5 min  | 
| MRPC (F1/Accuracy)            | Yes       | 92.45/89.71      | 88.85/83.82   | 91.13/87.75      | 91.13/87.50       | 0.5 min  | 
| QQP (F1/Accuracy)             | No        | 89.10/91.85      | 87.41/90.72   | 87.95/91.05      | 88.05/91.16       | 25 min   | 
| MNLI-m/mm (Accuracy)          | No        | 88.07/87.61      | 83.95/84.39   | 83.42/83.28      | 83.24/83.06       | 38 min   |
| QNLI (Accuracy)               | No        | 92.84            | 89.04         | 89.22            | 90.81             | 10 min   |
| RTE (Accuracy)                | No        | 79.42            | 61.01         | 67.15            | 61.37             | 0.5 min  |

```
mkdir ftlogs
for TASK in RTE CoLA SST-2 STS-B MRPC QQP MNLI QNLI; do
    paragen-run --config examples/bert/finetune/${TASK}.yaml --task.model.encoder.embed_layer_norm True> ftlogs/${TASK}.log 2>&1
done
```

# Using LightSeq to Speed up Training

1. Replace TransformerEncoder with LSTransformerEncoder at encoder's class;
2. Make a cup of white tea and enjoy it!
