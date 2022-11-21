#!/usr/bin/env bash

DATADIR=data
WOKESHOP=results

NAME=$1
MODEL=$2
DATA=$3
K=`expr $4 - 1`

echo "[NAME]    ${MODEL}_${NAME}"
echo "[DATA]    ${DATADIR}/${DATA}"
echo "[RESULT]  ${WOKESHOP}/${MODEL}_${NAME}"
echo "[CROSS]   K=$K"

# bash examples/antibody/scripts/individual.sh sars0510 ablang_heavy sars_germ_sm 10

if [ -f logs/${MODEL}_${NAME}_patient.log ]; then
    echo "Remove existing ${MODEL}_${NAME}_patient.log "
    rm logs/${MODEL}_${NAME}_patient.log
fi

for i in $(seq 0 $K)  
    do
        echo ${MODEL}_${NAME}_${i} 
        python3 examples/antibody/utils/analyze.py -mode GetBestIndividual \
                  -i ${WOKESHOP}/${MODEL}_${NAME}/${MODEL}_${NAME}_${i} \
                  -d ${DATADIR}/${DATA}/cross_valid_${i}_all.json >> logs/${MODEL}_${NAME}_patient.log
    done

python3 examples/antibody/utils/analyze.py -mode GetCrossIndividual -i logs/${MODEL}_${NAME}_patient.log




