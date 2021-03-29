#!/usr/bin/env bash

#python ./code/s4l_train.py --gpu 0,1 --n-labeled 10 --epochs 20 --val-iteration 1000 \
#	--data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 8 \
#	--pregenerated_data ./data/yahoo_answers_csv/training/ --mlm_batch_size 16 


python ./code/s4l_train.py --gpu 0,1 --n-labeled 10 --epochs 20 --val-iteration 1000 \
	--data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 8 \
	--pregenerated_data ./data/yahoo_answers_csv/training/ --mlm_batch_size 16 --resume ./ckpt/s4l/checkpoint.s4l 
