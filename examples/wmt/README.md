# WMT

[TOC]

## STEP 0. Installing Dependencies

```bash
bash s0_install_deps.sh
```

## STEP 1. Preparing Your Dataset

We firstly download the German-English corpus from [European Parliament Proceedings Parallel Corpus](http://statmt.org/europarl/) .

```bash
wget http://statmt.org/europarl/v7/de-en.tgz
tar zxvf de-en.tgz
mkdir -p data
mv europarl-v7.de-en.de data/train.de
mv europarl-v7.de-en.en data/train.en
```

## STEP 2. Cleaning the Dataset

### STEP 2.1 Sentence Normalization

In this step, we use `mosesdecoder` to normalize sentences (`tokenize->normalize->detokenize`) in the dataset. Besides, we also remove sentences that look invalid (empty sentences, too long sentences, and sentences with abnormal source-target ratio).

```bash
bash s2.1_norm_sents.sh
```

### STEP 2.2 Surface Level Filtering

In this step, we perform surface-level filtering to remove sentences which look invalid, such as:

- Data deduplication
- Delete parallel data with the same source and target
- Remove special tokens and unprintable tokens
- Remove HTML tags and inline URLs
- Remove words or characters that repeat more than 5 times
- Delete data that is too long( > 200 words), too short ( < 5 words), and the parallel data whose source and target lengths ratio are out of balance
- ...

We convert the dataset into the `jsonl` format (one json per line) so that we can better take a pair into consideration.

```bash
python3 tools/parallel2json.py \
    --src_file data/train.nor.de \
    --trg_file data/train.nor.en \
    --src_lang de \
    --trg_lang en \
    --output_file data/train.jsonl
```

The processed dataset looks like:
```json
{"src_text": "Wie Sie sicher aus der Presse und dem Fernsehen wissen, gab es in Sri Lanka mehrere Bombenexplosionen mit zahlreichen Toten.", "trg_text": "You will be aware from the press and television that there have been a number of bomb explosions and killings in Sri Lanka.", "src_lang": "de", "trg_lang": "en", "data_source": "train.nor.de", "monolingual": false, "pseudo": false, "preprocessed": false, "is_filtered": false, "filtered_func": null}
```

We then apply the surface filter to the dataset:

```bash
cat data/train.jsonl | python3 s22_surface_filter.py > data/train.22.jsonl
```

### STEP 2.3 Single Language Filtering

We filter the corpus by checking whether a sentence comes from a language (also known as language detection).

We use the [pycld3](https://pypi.org/project/pycld3) library to filter sentence pairs with the probability greater than 0.8 and the propotion greater than 60%. This scripts can also be used for filtering monolingual data.


```bash
python3 s23_lang_filter.py --fin data/train.22.jsonl --fout data/train.23.jsonl
```

Below is an example showing how `pycld3` works.

```python
>>> import cld3

>>> cld3.get_language("你好，世界")
LanguagePrediction(language='zh', probability=0.9994546175003052, is_reliable=True, proportion=1.0)

>>> for lang in cld3.get_frequent_languages(
...     "字节，say hello to world!",
...     num_langs=3
... ):
...     print(lang)
...
LanguagePrediction(language='en', probability=0.9840759634971619, is_reliable=True, proportion=0.7755101919174194)
LanguagePrediction(language='zh', probability=0.9988981485366821, is_reliable=True, proportion=0.22448979318141937)]
```


### STEP 2.4 Language Pair Filtering

#### Learning the Word Alignment

We use [fastalign](https://github.com/clab/fast_align) to automatically learn word alignment on the coarsely filtered corpus. To save your life, we provide a compiled version (`fast_align` and `atools`) in the `tools` directory. Input to `fastalign` must be tokenized and aligned into parallel sentences. Each line is a source language sentence and its target language translation, separated by a triple pipe symbol with leading and trailing white space (`|||`). Here is an example:

```
doch jetzt ist der Held gefallen . ||| but now the hero has fallen .
```

Below is the command for word alignment learning:

```bash
cat data/train.23.jsonl | python3 tools/align/json2fastalign.py > data/train.23.align
./tools/align/fast_align -i data/train.23.align    -d -v -o -p fwd_params >fwd_align 2>fwd_err
./tools/align/fast_align -i data/train.23.align -r -d -v -o -p rev_params >rev_align 2>rev_err
./tools/align/atools -i fwd_align -j rev_align -c grow-diag-final-and >data/align.merged
rm fwd_* rev*
```

We will get something like this, which means the `i`th word in the source language is aligned with the `j`th word in the target language:

```
0-0 1-1 2-4 3-2 4-3 5-5 6-6
```

#### Building the Bilingual Vocabulary

After that, we extract a bilingual vocabulary from the learned word alignments:

```bash
python3 tools/json2parallel.py \
    --input_filename data/train.23.jsonl \
    --src_filename data/train.23.de \
    --trg_filename data/train.23.en 

python3 tools/align/extract_bilingual_vocabulary.py \
    --overwrite \
    --src_filename data/train.23.de \
    --trg_filename data/train.23.en \
    --align_filename data/align.merged \
    --dict_filename data/dict --threshold 0.01 --ignore_gap 30 

mv data/dict.src2trg data/dict.de2en
mv data/dict.trg2src data/dict.en2de

python3 tools/align/json2db.py --lang_pair de2en --wd data

rm data/dict.* data/align.merged*
```

The generated `dict.*2*` looks like this, which means the probability that a word in the source language is aligned with another in the target one. We will convert it to `.db` file for later use.

```json
{"die": {"the": 0.577, "which": 0.071, "to": 0.034, "that": 0.016, "of": 0.015, "for": 0.012}}
```

#### Filtering by Sentence Alignment Score

Finally, we use the word alignment score to compute the sentence alignment score, and filter invalid sentence pairs with a given threshold.

```bash
python3 tools/align/extract_corpus.py \
    -p de2en \
    -f1 train.23.de \
    -f2 train.23.en \
    -wd data \
    -o train.24.jsonl

python3 tools/json2parallel.py \
    --input_filename data/train.24.jsonl \
    --src_filename data/train.24.de \
    --trg_filename data/train.24.en 
```

The generated `train.24.de` and `train.24.en` are parallel data for training.


## STEP 3. Training A Strong Auto-Regressive Teacher

In this step, we train a strong auto-regressive teacher which can achieve high performance. Here are two suggested architechtures: `12e12d1024` and `16e6d1024`. `12e12d1024` means that the Transformer model has 12 encoder layers and 12 decoder layers, with the hidden size to be 1024.

We use the model to generate translations for the source lanugage in the training data. The KD data (known as `Knowledge Distillation`) are used for training a student model  since it is well-known that there are less multi-modality in the KD data, which can benefit the NAT model's training process.

## STEP 4: Training A Smart Non-AutoRegressive Student

In this step, we train an non-autoregressive model (GLAT) on the KD data. We suggest you use the `16e6d1024` architechture. Adding a `DLCL` component can make the training process more stable.

## STEP 4: ReRanking

In this step, we re-rank the nbest list generated by translation model. The rerank hypothses based on multiple pre-defined features and tune the feature weights on dev set with `kb-mira `algorithm.

We suggest that you refer to `tools/rescore/example` for details.

Here we provide some scripts for ranking your hypothses, whose data format looks like:

```
<hypothsis 1 of sentence 1>
<hypothsis 2 of sentence 1>
<hypothsis 3 of sentence 1>
<hypothsis 1 of sentence 2>
<hypothsis 2 of sentence 2>
<hypothsis 3 of sentence 2>
...
```

We can score the sentences with several features, such as `chrf`, `selfbelu`, `neural language model`, `statistic language model`, etc. Here are some scripts:

```bash
# Since these scores are computed within a batch of hypothses, you should specify the number of candidates.
python3 tools/scorer/chrf.py     --hypo_filename <hypo> --out_filename <out> --num_candidates <num>
python3 tools/scorer/selfbleu.py --hypo_filename <hypo> --out_filename <out> --num_candidates <num>

# Score by neural language model requires installation of fairseq.
# Score by neural language model requires installation of kenlm.
python3 tools/scorer/nlm.py      --hypo_filename <hypo> --out_filename <out>
python3 tools/scorer/slm.py      --hypo_filename <hypo> --out_filename <out>

# Score by reference, this is a cheating feature just for debugging your reranker.
python3 tools/scorer/refbleu.py  --hypo_filename <hypo> --ref_filename <ref> --out_filename <out>
```

You can also add the scores of AT models for reranking, which is supposed to gain a great improvement.

