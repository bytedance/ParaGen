#!/usr/bin/env bash

HDFSPRE="hdfs_workdir"

MODE=$1
disease=$2
models=$3
NAME=$4

# bash examples/antibody/scripts/disease_diagnosis.sh [download/seq/ind] sars eatlm name

if [ "$MODE" == "download" ]; then
  for d in $disease  
    do
      for m in $models
      do
        echo "Download ${m}_${d}${NAME}.log"
        hdfs dfs -get $HDFSPRE/logs/${m}_${d}${NAME}.log logs/
        echo "Download result ${m}_${d}${NAME}"
        hdfs dfs -get $HDFSPRE/results/${m}_${d}${NAME} results/
      done
    done

elif [ "$MODE" == "seq" ]; then
  for d in $disease  
    do
      for m in $models
        do
          echo "Sequence-level ${m}_${d}${NAME}.log"
          python3 examples/antibody/analyze.py -mode GetCrossResult -i logs/${m}_${d}${NAME}.log
        done
    done

elif [ "$MODE" == "ind" ]; then
  for d in $disease  
    do
      for m in $models
        do
          echo "Individual-level ${m}_${d}${NAME}.log"
          bash individual.sh ${d}${NAME} ${m} ${d}_germ_sm_3 5
        done
    done
fi

