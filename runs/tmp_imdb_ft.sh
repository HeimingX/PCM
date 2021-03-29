#!/usr/bin/env bash

LBS=4
UBS=8

T=1
DATASET=./data/imdb_csv/

LR_MAIN=5e-6
LR_LAST=5e-4
CONFID=0.95

# ============================== uda probing words: start from class names ==============================
PROB_WORD_TYPE=dynamic
PROB_NUM=30
#PROB_NUM=10
LOSS_FUNC=bce
PROB_CONFID=0.7

SP_NAME=_1infer_ft_update_wBstAcc_run2
SAVE_SP=_1infer_ft_update_pn30_wBstAcc_run2

GPU=0,1

NLAB=10
PROB_TOPK=30
sleep 4h
NLAB=3
BEST_ACC=0.608
PROB_FILE_NAME=ft_nlab3_top30_ep19_wCLS.npy
python ./code/uda_with_probing_words_1vsall_pl1st.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
	--lambda-u 1 --T $T --confid $CONFID --prob_loss_func $LOSS_FUNC \
	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP
