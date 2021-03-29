#!/usr/bin/env bash

# 10 classes
DATASET=./data/dbpedia_csv/

GPU=1

#LR_MAIN=1e-5
#LR_LAST=1e-3

LR_MAIN=5e-6
LR_LAST=5e-4

BS=8

#default: use_gap, not_use_cls

# =================== w/ CLS & SEP; GAP w/o CLS ==========================
GPU=1
NLAB=10
SP=_GAPnoCLS
python ./code/normal_train.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
  --data-path $DATASET --batch-size $BS --epochs 20 \
  --lrmain $LR_MAIN --lrlast $LR_LAST --specific_name $SP

NLAB=5
python ./code/normal_train.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
  --data-path $DATASET --batch-size $BS --epochs 20 \
  --lrmain $LR_MAIN --lrlast $LR_LAST --specific_name $SP

NLAB=3
python ./code/normal_train.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
  --data-path $DATASET --batch-size $BS --epochs 20 \
  --lrmain $LR_MAIN --lrlast $LR_LAST --specific_name $SP