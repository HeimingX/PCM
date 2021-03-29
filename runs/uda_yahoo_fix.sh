#!/usr/bin/env bash

LBS=4
UBS=8
DATASET=./data/yahoo_answers_csv/yahoo_answers_csv/
LR_MAIN=5e-6
LR_LAST=5e-4

T=0.5
CONFID=0.95

# ============================== dynamic probing words: start from class names ========================
PROB_FILE_NAME=cls_names
PROB_WORD_TYPE=ClsNames
PROB_NUM=75
LOSS_FUNC=bce
PROB_CONFID=0.7

SP_NAME=_1infer_run2
SAVE_SP=_1infer_clsName_update_pn10

GPU=1

NLAB=5
PROB_TOPK=30
python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval