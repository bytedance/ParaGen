lang=$1
finput=$2
vocab_size=$3
name=${lang}.sp

spm_train \
    --input="${finput}" \
    --model_prefix="${name}" \
    --vocab_size=$vocab_size \
    --unk_id=3 \
    --bos_id=0 \
    --eos_id=2 \
    --pad_id=1 \
    --model_type=unigram \
    --num_threads=32 \
    --character_coverage=0.999999 \
    --input_sentence_size=10000000 \
    --shuffle_input_sentence=true 
    > ${name}.log 2>&1

sed -i -E 's/\.[0-9]+$//g' ${name}.vocab
