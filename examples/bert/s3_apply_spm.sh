spm_model=$1
path=.
finput=$2
foutput=$3
nproc=${4:-40}

split -n l/${nproc} --numeric-suffixes=1 ${path}/${finput} ${path}/_TMP_

for part in ${path}/_TMP_*; do
    echo Start encoding $part...
    spm_encode \
        --model=$spm_model \
        --input=$part \
        --output=$part.OUT &
done
wait

cat ${path}/_TMP_*.OUT > ${path}/${foutput}
rm ${path}/_TMP_*
