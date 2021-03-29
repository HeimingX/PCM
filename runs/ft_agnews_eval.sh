#!/usr/bin/env bash

# 4 classes
DATASET=./data/ag_news_csv/

GPU=0,1

#LR_MAIN=1e-5
#LR_LAST=1e-3

LR_MAIN=5e-6
LR_LAST=5e-4

BS=8
#NLAB=10
#PROB_TOPK=5
##RESUME=./experiments/fine_tune/ag_news_10/ckpt/best.lr1e-05_0.001_ep20_bs8_noClsSep_wGap_seed0.ckpt
###RESUME=./experiments/fine_tune/ag_news_10/ckpt/best.lr5e-06_0.0005_ep20_bs8_noClsSep_wGap_seed0.ckpt
##python ./code/normal_train.py --gpu $GPU --n-labeled $NLAB --data-path $DATASET --batch-size $BS --epochs 20 --lrmain $LR_MAIN --lrlast $LR_LAST --resume $RESUME --eval --prob_topk $PROB_TOPK
#
#NLAB=5
#RESUME=./experiments/fine_tune/ag_news_5/ckpt/best.lr1e-05_0.001_ep20_bs8_noClsSep_wGap_seed0.ckpt
##RESUME=./experiments/fine_tune/ag_news_5/ckpt/best.lr5e-06_0.0005_ep20_bs8_noClsSep_wGap_seed0.ckpt
#python ./code/normal_train.py --gpu $GPU --n-labeled $NLAB --data-path $DATASET --batch-size $BS --epochs 20 --lrmain $LR_MAIN --lrlast $LR_LAST --resume $RESUME --eval --prob_topk $PROB_TOPK
#
#NLAB=4
#RESUME=./experiments/fine_tune/ag_news_4/ckpt/best.lr1e-05_0.001_ep20_bs8_noClsSep_wGap_seed0.ckpt
##RESUME=./experiments/fine_tune/ag_news_4/ckpt/best.lr5e-06_0.0005_ep20_bs8_noClsSep_wGap_seed0.ckpt
#python ./code/normal_train.py --gpu $GPU --n-labeled $NLAB --data-path $DATASET --batch-size $BS --epochs 20 --lrmain $LR_MAIN --lrlast $LR_LAST --resume $RESUME --eval --prob_topk $PROB_TOPK
#
#NLAB=3
##SP_NAME=_run2
#RESUME=./experiments/fine_tune/ag_news_3/ckpt/best.lr1e-05_0.001_ep20_bs8_noClsSep_wGap_seed0.ckpt
##RESUME=./experiments/fine_tune/ag_news_3/ckpt/best.lr5e-06_0.0005_ep20_bs8_noClsSep_wGap_seed0.ckpt
#python ./code/normal_train.py --gpu $GPU --n-labeled $NLAB --data-path $DATASET --batch-size $BS --epochs 20 --lrmain $LR_MAIN --lrlast $LR_LAST --resume $RESUME --eval --prob_topk $PROB_TOPK
##  --specific_name $SP_NAME
#
#NLAB=2
#RESUME=./experiments/fine_tune/ag_news_2/ckpt/best.lr1e-05_0.001_ep20_bs8_noClsSep_wGap_seed0.ckpt
##RESUME=./experiments/fine_tune/ag_news_2/ckpt/best.lr5e-06_0.0005_ep20_bs8_noClsSep_wGap_seed0.ckpt
#python ./code/normal_train.py --gpu $GPU --n-labeled $NLAB --data-path $DATASET --batch-size $BS --epochs 20 --lrmain $LR_MAIN --lrlast $LR_LAST --resume $RESUME --eval --prob_topk $PROB_TOPK

# ---------- with cls & sep token ----------
NLAB=10
PSP=_wCLS
PROB_TOPK=30
RESUME=./experiments/fine_tune/ag_news_${NLAB}/ckpt/best.lr5e-06_0.0005_ep20_bs8_wClsSep_wGap_seed0.ckpt
python ./code/normal_train.py --gpu $GPU --n-labeled $NLAB --add-cls-sep --data-path $DATASET --batch-size $BS --epochs 20 --lrmain $LR_MAIN --lrlast $LR_LAST --resume $RESUME --eval --prob_topk $PROB_TOPK --prob_save_sp $PSP

NLAB=5
RESUME=./experiments/fine_tune/ag_news_${NLAB}/ckpt/best.lr5e-06_0.0005_ep20_bs8_wClsSep_wGap_seed0.ckpt
python ./code/normal_train.py --gpu $GPU --n-labeled $NLAB --add-cls-sep --data-path $DATASET --batch-size $BS --epochs 20 --lrmain $LR_MAIN --lrlast $LR_LAST --resume $RESUME --eval --prob_topk $PROB_TOPK --prob_save_sp $PSP

#NLAB=4
#RESUME=./experiments/fine_tune/ag_news_${NLAB}/ckpt/best.lr5e-06_0.0005_ep20_bs8_wClsSep_wGap_seed0.ckpt
#python ./code/normal_train.py --gpu $GPU --n-labeled $NLAB --add-cls-sep --data-path $DATASET --batch-size $BS --epochs 20 --lrmain $LR_MAIN --lrlast $LR_LAST --resume $RESUME --eval --prob_topk $PROB_TOPK --prob_save_sp $PSP

NLAB=3
RESUME=./experiments/fine_tune/ag_news_${NLAB}/ckpt/best.lr5e-06_0.0005_ep20_bs8_wClsSep_wGap_seed0.ckpt
python ./code/normal_train.py --gpu $GPU --n-labeled $NLAB --add-cls-sep --data-path $DATASET --batch-size $BS --epochs 20 --lrmain $LR_MAIN --lrlast $LR_LAST --resume $RESUME --eval --prob_topk $PROB_TOPK --prob_save_sp $PSP

#NLAB=2
#RESUME=./experiments/fine_tune/ag_news_${NLAB}/ckpt/best.lr5e-06_0.0005_ep20_bs8_wClsSep_wGap_seed0.ckpt
#python ./code/normal_train.py --gpu $GPU --n-labeled $NLAB --add-cls-sep --data-path $DATASET --batch-size $BS --epochs 20 --lrmain $LR_MAIN --lrlast $LR_LAST --resume $RESUME --eval --prob_topk $PROB_TOPK --prob_save_sp $PSP