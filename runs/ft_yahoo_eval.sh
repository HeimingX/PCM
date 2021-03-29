#!/usr/bin/env bash

# 10 classes
DATASET=./data/yahoo_answers_csv/yahoo_answers_csv/

GPU=3

#LR_MAIN=1e-5
#LR_LAST=1e-3

LR_MAIN=5e-6
LR_LAST=5e-4

BS=8
NLAB=10
PROB_TOPK=30
##RESUME=./experiments/fine_tune/yahoo_answers_10/ckpt/best.lr1e-05_0.001_ep20_bs8_noClsSep_wGap_seed0.ckpt
#RESUME=./experiments/fine_tune/yahoo_answers_10/ckpt/best.lr5e-06_0.0005_ep20_bs8_noClsSep_wGap_seed0.ckpt
#python ./code/normal_train.py --gpu $GPU --n-labeled $NLAB --data-path $DATASET --batch-size $BS --epochs 20 --lrmain $LR_MAIN --lrlast $LR_LAST --resume $RESUME --eval --prob_topk $PROB_TOPK
#
#NLAB=5
##RESUME=./experiments/fine_tune/yahoo_answers_5/ckpt/best.lr1e-05_0.001_ep20_bs8_noClsSep_wGap_seed0.ckpt
#RESUME=./experiments/fine_tune/yahoo_answers_5/ckpt/best.lr5e-06_0.0005_ep20_bs8_noClsSep_wGap_seed0.ckpt
#python ./code/normal_train.py --gpu $GPU --n-labeled $NLAB --data-path $DATASET --batch-size $BS --epochs 20 --lrmain $LR_MAIN --lrlast $LR_LAST --resume $RESUME --eval --prob_topk $PROB_TOPK
#
#NLAB=4
##RESUME=./experiments/fine_tune/yahoo_answers_4/ckpt/best.lr1e-05_0.001_ep20_bs8_noClsSep_wGap_seed0.ckpt
#RESUME=./experiments/fine_tune/yahoo_answers_4/ckpt/best.lr5e-06_0.0005_ep20_bs8_noClsSep_wGap_seed0.ckpt
#python ./code/normal_train.py --gpu $GPU --n-labeled $NLAB --data-path $DATASET --batch-size $BS --epochs 20 --lrmain $LR_MAIN --lrlast $LR_LAST --resume $RESUME --eval --prob_topk $PROB_TOPK
#
#NLAB=3
##RESUME=./experiments/fine_tune/yahoo_answers_3/ckpt/best.lr1e-05_0.001_ep20_bs8_noClsSep_wGap_seed0.ckpt
#RESUME=./experiments/fine_tune/yahoo_answers_3/ckpt/best.lr5e-06_0.0005_ep20_bs8_noClsSep_wGap_seed0.ckpt
#python ./code/normal_train.py --gpu $GPU --n-labeled $NLAB --data-path $DATASET --batch-size $BS --epochs 20 --lrmain $LR_MAIN --lrlast $LR_LAST --resume $RESUME --eval --prob_topk $PROB_TOPK
#
#NLAB=2
##RESUME=./experiments/fine_tune/yahoo_answers_2/ckpt/best.lr1e-05_0.001_ep20_bs8_noClsSep_wGap_seed0.ckpt
#RESUME=./experiments/fine_tune/yahoo_answers_2/ckpt/best.lr5e-06_0.0005_ep20_bs8_noClsSep_wGap_seed0.ckpt
#python ./code/normal_train.py --gpu $GPU --n-labeled $NLAB --data-path $DATASET --batch-size $BS --epochs 20 --lrmain $LR_MAIN --lrlast $LR_LAST --resume $RESUME --eval --prob_topk $PROB_TOPK


# ---------- with cls & sep token ----------
NLAB=10
PSP=_wCLS
RESUME=./experiments/fine_tune/yahoo_answers_${NLAB}/ckpt/best.lr5e-06_0.0005_ep20_bs8_wClsSep_wGap_seed0.ckpt
python ./code/normal_train.py --gpu $GPU --n-labeled $NLAB --add-cls-sep --data-path $DATASET --batch-size $BS --epochs 20 --lrmain $LR_MAIN --lrlast $LR_LAST --resume $RESUME --eval --prob_topk $PROB_TOPK --prob_save_sp $PSP

NLAB=5
RESUME=./experiments/fine_tune/yahoo_answers_${NLAB}/ckpt/best.lr5e-06_0.0005_ep20_bs8_wClsSep_wGap_seed0.ckpt
python ./code/normal_train.py --gpu $GPU --n-labeled $NLAB --add-cls-sep --data-path $DATASET --batch-size $BS --epochs 20 --lrmain $LR_MAIN --lrlast $LR_LAST --resume $RESUME --eval --prob_topk $PROB_TOPK --prob_save_sp $PSP

NLAB=4
RESUME=./experiments/fine_tune/yahoo_answers_${NLAB}/ckpt/best.lr5e-06_0.0005_ep20_bs8_wClsSep_wGap_seed0.ckpt
python ./code/normal_train.py --gpu $GPU --n-labeled $NLAB --add-cls-sep --data-path $DATASET --batch-size $BS --epochs 20 --lrmain $LR_MAIN --lrlast $LR_LAST --resume $RESUME --eval --prob_topk $PROB_TOPK --prob_save_sp $PSP

NLAB=3
RESUME=./experiments/fine_tune/yahoo_answers_${NLAB}/ckpt/best.lr5e-06_0.0005_ep20_bs8_wClsSep_wGap_seed0.ckpt
python ./code/normal_train.py --gpu $GPU --n-labeled $NLAB --add-cls-sep --data-path $DATASET --batch-size $BS --epochs 20 --lrmain $LR_MAIN --lrlast $LR_LAST --resume $RESUME --eval --prob_topk $PROB_TOPK --prob_save_sp $PSP

NLAB=2
RESUME=./experiments/fine_tune/yahoo_answers_${NLAB}/ckpt/best.lr5e-06_0.0005_ep20_bs8_wClsSep_wGap_seed0.ckpt
python ./code/normal_train.py --gpu $GPU --n-labeled $NLAB --add-cls-sep --data-path $DATASET --batch-size $BS --epochs 20 --lrmain $LR_MAIN --lrlast $LR_LAST --resume $RESUME --eval --prob_topk $PROB_TOPK --prob_save_sp $PSP