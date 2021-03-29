#!/usr/bin/env bash

LBS=4
UBS=8

DATASET=./data/ag_news_csv/
T=1
# ================ original uda ========================
GPU=2

NLAB=10
#python ./code/uda.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _gapFeaOnly #--eval --resume ./experiments/uda/
NLAB=5
#python ./code/uda.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _gapFeaOnly #--eval --resume ./experiments/uda/
NLAB=4
#python ./code/uda.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _gapFeaOnly #--eval --resume ./experiments/uda/
NLAB=3
#python ./code/uda.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _gapFeaOnly #--eval --resume ./experiments/uda/
NLAB=2
python ./code/uda.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
	--lambda-u 1 --T $T \
	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _gapFeaOnly #--eval --resume ./experiments/uda/
NLAB=1
#python ./code/uda.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _gapFeaOnly #--eval --resume ./experiments/uda/
