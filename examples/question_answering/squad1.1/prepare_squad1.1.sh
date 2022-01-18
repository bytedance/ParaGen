#!/usr/bin/env bash

curl https://worksheets.codalab.org/rest/bundles/0x7e0a0a21057c4d989aa68da42886ceb9/contents/blob/ > train-v1.1.json
curl https://worksheets.codalab.org/rest/bundles/0x8f29fe78ffe545128caccab74eb06c57/contents/blob/ > dev-v1.1.json
curl https://worksheets.codalab.org/rest/bundles/0x4c6febb3f9574587a6729b23b5e2f290/contents/blob/ > eval.py


python3 prepare_squad1.1.py
