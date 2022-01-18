#!/usr/bin/env bash

mosesdecoder=./mosesdecoder

decodes_file=$1
# use original newstest.txt for eval
gold_targets=$2

python eval.py --gold $gold_targets --hypo $decodes_file

# Put compounds in ATAT format (comparable to papers like GNMT, ConvS2S).
# See https://nlp.stanford.edu/projects/nmt/ :
# 'Also, for historical reasons, we split compound words, e.g.,
#    "rich-text format" --> rich ##AT##-##AT## text format."'
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $gold_targets.tok > $gold_targets.tok.atat
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $decodes_file.tok > $decodes_file.tok.atat

# Get BLEU.
perl $mosesdecoder/scripts/generic/multi-bleu.perl $gold_targets.tok.atat < $decodes_file.tok.atat
