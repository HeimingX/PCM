#!/usr/bin/env bash

LBS=4
UBS=8

T=0.5
DATASET=./data/yahoo_answers_csv/yahoo_answers_csv/

LR_MAIN=5e-6
LR_LAST=5e-4

PROB_WORD_TYPE=dynamic
PROB_NUM=75
LOSS_FUNC=bce
CONFID=0.95
PROB_CONFID=0.7

SP_NAME=_ft
SAVE_SP=_ft

GPU=2
PROB_TOPK=30

NLAB=10
PROB_FILE_NAME=ft_nlab10_top30_ep9.npy
#PROB_FILE_NAME=uda_dyn_nlab10_top30.0_noStopWords_probWrods_ep0_ft.npy
#RESUME=./experiments/uda_probing_words/yahoo_answers_10/ckpt/best.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromUDA_dynamic_parallel_prbNum75_wPrbWrd1.0_prbLossbce_cnfdPrb0.7_seed0_ft.ckpt
#NLAB=5
#PROB_FILE_NAME=
#NLAB=4
#PROB_FILE_NAME=
#NLAB=3
#PROB_FILE_NAME=
#NLAB=2
#PROB_FILE_NAME=

python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume $RESUME #--eval

#	--local_machine

