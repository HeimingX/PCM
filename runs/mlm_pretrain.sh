#!/usr/bin/env bash

python ./code/mlm_pretrain.py --gpu 2 --epochs 20 --val-iteration 1000 \
	--pregenerated_data ./data/yahoo_answers_csv/training/ --mlm_batch_size 16 
