#!/usr/bin/env bash

LBS=4
UBS=8
DATASET=./data/yahoo_answers_csv/yahoo_answers_csv/
LR_MAIN=5e-6
LR_LAST=5e-4

T=0.5
CONFID=0.95
SEEDL=2

# ================ original uda ========================
GPU=1

#NLAB=10
#python ./code/uda.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --seed_l $SEEDL \
#	--lambda-u 1 --T $T \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _gapFeaOnly #--eval --resume ./experiments/uda/

#NLAB=5
#python ./code/uda.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --seed_l $SEEDL \
#	--lambda-u 1 --T $T \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _gapFeaOnly #--eval --resume ./experiments/uda/

#GPU=2
#NLAB=3
#python ./code/uda.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --seed_l $SEEDL \
#	--lambda-u 1 --T $T \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _gapFeaOnly #--eval --resume ./experiments/uda/


# ------- update probing words with consistent ratio on unlabeled val set --------
GPU=2

PROB_FILE_NAME=cls_names
PROB_WORD_TYPE=dynamic
LOSS_FUNC=bce
#PROB_CONFID=0.7
PROB_CONFID=0.8
PROB_TOPK=30

#PROB_NUM=75
#SP_NAME=_consRatio
#SAVE_SP=_clsName_pn${PROB_NUM}_consRatio_sdl${SEEDL}
#NLAB=5
#python ./code/uda_with_probing_words_1vsall_consist.py --gpu $GPU --n-labeled $NLAB --add-cls-sep --local_machin \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --seed_l $SEEDL \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval

PROB_NUM=5
SP_NAME=_consRatio
SAVE_SP=_clsName_pn${PROB_NUM}_pc${PROB_CONFID}_consRatio_sdl${SEEDL}
NLAB=3
python ./code/uda_with_probing_words_1vsall_consist.py --gpu $GPU --n-labeled $NLAB --add-cls-sep --local_machin \
	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --seed_l $SEEDL \
	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval
