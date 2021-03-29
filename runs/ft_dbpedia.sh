#!/usr/bin/env bash

# 14 classes
DATASET=./data/dbpedia_csv/

GPU=2

#LR_MAIN=1e-5
#LR_LAST=1e-3

LR_MAIN=5e-6
LR_LAST=5e-4

#BS=8
#NLAB=10
#python ./code/normal_train.py --gpu $GPU --n-labeled $NLAB --data-path $DATASET --batch-size $BS --epochs 20 --lrmain $LR_MAIN --lrlast $LR_LAST
#
#NLAB=5
#python ./code/normal_train.py --gpu $GPU --n-labeled $NLAB --data-path $DATASET --batch-size $BS --epochs 20 --lrmain $LR_MAIN --lrlast $LR_LAST
#
#NLAB=4
#python ./code/normal_train.py --gpu $GPU --n-labeled $NLAB --data-path $DATASET --batch-size $BS --epochs 20 --lrmain $LR_MAIN --lrlast $LR_LAST
#
#NLAB=3
#python ./code/normal_train.py --gpu $GPU --n-labeled $NLAB --data-path $DATASET --batch-size $BS --epochs 20 --lrmain $LR_MAIN --lrlast $LR_LAST
#
#NLAB=2
#python ./code/normal_train.py --gpu $GPU --n-labeled $NLAB --data-path $DATASET --batch-size $BS --epochs 20 --lrmain $LR_MAIN --lrlast $LR_LAST

BS=8
NLAB=10
python ./code/normal_train.py --gpu $GPU --n-labeled $NLAB  --add-cls-sep \
  --data-path $DATASET --batch-size $BS --epochs 20 \
  --lrmain $LR_MAIN --lrlast $LR_LAST

NLAB=5
python ./code/normal_train.py --gpu $GPU --n-labeled $NLAB  --add-cls-sep \
  --data-path $DATASET --batch-size $BS --epochs 20 \
  --lrmain $LR_MAIN --lrlast $LR_LAST

NLAB=4
python ./code/normal_train.py --gpu $GPU --n-labeled $NLAB  --add-cls-sep \
  --data-path $DATASET --batch-size $BS --epochs 20 \
  --lrmain $LR_MAIN --lrlast $LR_LAST

NLAB=3
python ./code/normal_train.py --gpu $GPU --n-labeled $NLAB  --add-cls-sep \
  --data-path $DATASET --batch-size $BS --epochs 20 \
  --lrmain $LR_MAIN --lrlast $LR_LAST

NLAB=2
python ./code/normal_train.py --gpu $GPU --n-labeled $NLAB  --add-cls-sep \
  --data-path $DATASET --batch-size $BS --epochs 20 \
  --lrmain $LR_MAIN --lrlast $LR_LAST