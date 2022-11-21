#!/usr/bin/env bash

DATADIR="hdfs_datadir"
WOKESHOP="hdfs_workdir"

CONFIG=$1
NAME=$2
MODEL=$3
DATA=$4
K=`expr $5 - 1`
LOG=$6

argslist=""
for (( i = 7; i <= $# ; i++ ))
  do
    j=${!i}
    argslist="${argslist} $j "
  done

echo "[CONFIG]  ${WOKESHOP}/configs/${CONFIG}.yaml" 
echo "[NAME]    ${MODEL}_${NAME}"
echo "[MODEL]   ${WOKESHOP}/models/${MODEL}/best.pt"
echo "[DATA]    ${DATADIR}/${DATA}"
echo "[CROSS]   K=$K"
echo "[ARGS]    $argslist" 

# bash examples/antibody/scripts/cross_valid.sh paratope_germ paratope_cross eatlm paratope_proc 10 [save/drop]

python3 -m pip install -U -e .

if [ "$LOG" == "save" ]; then
  echo "Need to save the results!"
  for i in $(seq 0 $K)  
    do
        echo ${MODEL}_${i} 
        paragen-run --config ${WOKESHOP}/configs/${CONFIG}.yaml \
                    --lib examples/antibody/src \
                    --task.data.train.path ${DATADIR}/${DATA}/cross_train_${i}_all.json \
                    --task.data.valid.path ${DATADIR}/${DATA}/cross_valid_${i}_all.json \
                    --task.model.path ${WOKESHOP}/models/${MODEL}/best.pt \
                    --task.evaluator.save_hypo_dir results/${MODEL}_${NAME}_${i} \
                    $argslist &>> result.log
    done

    echo "upload results to ${WOKESHOP}/results/${MODEL}_${NAME}"
    hdfs dfs -mkdir  ${WOKESHOP}/results/${MODEL}_${NAME}
    hdfs dfs -put -f results/* ${WOKESHOP}/results/${MODEL}_${NAME}

elif [ "$LOG" == "drop" ]; then
  echo "Do not save the results!"
  for i in $(seq 0 $K)  
      do
          echo ${MODEL}_${i} 
          paragen-run --config ${WOKESHOP}/configs/${CONFIG}.yaml \
                      --lib examples/antibody/src \
                      --task.data.train.path ${DATADIR}/${DATA}/cross_train_${i}_all.json \
                      --task.data.valid.path ${DATADIR}/${DATA}/cross_valid_${i}_all.json \
                      --task.model.path ${WOKESHOP}/models/${MODEL}/best.pt \
                      --task.trainer.save_model_dir None \
                      $argslist &>> result.log
      done
fi


echo "upload result.log to ${WOKESHOP}/logs/${MODEL}_${NAME}.log"
hdfs dfs -put -f result.log ${WOKESHOP}/logs/${MODEL}_${NAME}.log




