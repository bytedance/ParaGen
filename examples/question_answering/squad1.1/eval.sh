#!/usr/bin/env bash

until [[ -z "$1" ]]
do
    case $1 in
        -hypo)
            shift; hypo=$1;
            shift;;
    esac
done

python3 merge.py --hypo ${hypo} --id dev.id --out dev.hypo
python2 eval.py dev-v1.1.json dev.hypo