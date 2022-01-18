#!/usr/bin/env bash

wget http://dl.fbaipublicfiles.com/nat/original_dataset.zip
unzip original_dataset.zip

mv wmt14_ende data

mkdir resources

paragen-build-tokenizer --config configs/build_tokenizer.yaml
