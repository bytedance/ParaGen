mosesdir=mosesdecoder/scripts
lang=$1
finput=$2
foutput=$3

cat $finput \
| perl $mosesdir/tokenizer/normalize-punctuation.perl -l $lang \
| perl $mosesdir/tokenizer/tokenizer.perl -a -threads 50 -l $lang \
> $foutput
