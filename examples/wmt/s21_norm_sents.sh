# suffix of source language files
SRC=de
# suffix of target language files
TRG=en
# prefix for files
PREFIX=train
# $DATA path
DATA=data

# path to moses decoder
mosesdecoder=mosesdecoder


# tokenize
cat $DATA/$PREFIX.$SRC | \
$mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $SRC -threads 50 | \
$mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $SRC -threads 50 > $DATA/$PREFIX.tok.$SRC

cat $DATA/$PREFIX.$TRG | \
$mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $TRG -threads 50 | \
$mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $TRG -threads 50 > $DATA/$PREFIX.tok.$TRG


# clean empty and long sentences, and sentences with high source-target ratio (training corpus only)
if [ "$PREFIX" = train ] ; then
    $mosesdecoder/scripts/training/clean-corpus-n.perl -ratio 1.5 $DATA/$PREFIX.tok $SRC $TRG $DATA/$PREFIX.tok.clean 1 175
    cat $DATA/$PREFIX.tok.clean.$SRC | \
    $mosesdecoder/scripts/tokenizer/detokenizer.perl -a  -l $SRC -threads 50 > $DATA/$PREFIX.nor.$SRC
    cat $DATA/$PREFIX.tok.clean.$TRG | \
    $mosesdecoder/scripts/tokenizer/detokenizer.perl -a  -l $TRG -threads 50 > $DATA/$PREFIX.nor.$TRG
    echo 'clean DATA by ratio'
else
    cat $DATA/$PREFIX.tok.$SRC | \
    $mosesdecoder/scripts/tokenizer/detokenizer.perl -a  -l $SRC -threads 50 > $DATA/$PREFIX.nor.$SRC
    cat $DATA/$PREFIX.tok.$TRG | \
    $mosesdecoder/scripts/tokenizer/detokenizer.perl -a  -l $TRG -threads 50 > $DATA/$PREFIX.nor.$TRG
    echo 'clean DATA by ratio'
fi
