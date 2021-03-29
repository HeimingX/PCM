#!/usr/bin/env bash

LBS=4
UBS=8

DATASET=./data/ag_news_csv/
T=1

LR_MAIN=5e-6
LR_LAST=5e-4
CONFID=0.95

# ============================== uda probing words: start from class names ==============================
PROB_WORD_TYPE=dynamic
PROB_NUM=75
LOSS_FUNC=bce
PROB_CONFID=0.7
PROB_TOPK=30

#SP_NAME=_pl1st_ft_update_wBstAcc
#SAVE_SP=_pl1st_ft_update_pn75_wBstAcc

GPU=0,1

#NLAB=10
#BEST_ACC=0.8504
#PROB_FILE_NAME=ft_nlab10_top30_ep7_wCLS.npy
#python ./code/uda_with_probing_words_1vsall_pl1st_agnews.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ag_news_10/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T1.0_confid0.95_prbFromFt_dynamic_parallel_prbNum75_wPrbWrd1.0_prbLossbce_cnfdPrb0.7_seed0_pl1st_ft_update_wBstAcc.ckpt

#SP_NAME=_noPl1st_ft_update_wBstAcc
#SAVE_SP=_noPl1st_ft_update_pn75_wBstAcc
#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ag_news_10/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T1.0_confid0.95_prbFromFt_dynamic_parallel_prbNum75_wPrbWrd1.0_prbLossbce_cnfdPrb0.7_seed0_pl1st_ft_update_wBstAcc.ckpt

#NLAB=5
#BEST_ACC=0.7925
#PROB_FILE_NAME=ft_nlab5_top30_ep6_wCLS.npy
#python ./code/uda_with_probing_words_1vsall_pl1st_agnews.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP

NLAB=3

#PROB_NUM=30
#SP_NAME=_pl1st_ft_update_pn${PROB_NUM}_wBstAcc
#SAVE_SP=_pl1st_ft_update_pn${PROB_NUM}_wBstAcc
#
#BEST_ACC=0.7088
#PROB_FILE_NAME=ft_nlab3_top30_ep10_wCLS.npy
#python ./code/uda_with_probing_words_1vsall_pl1st_agnews.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP


#PROB_NUM=75
#SP_NAME=_pl1st_ft_update_pn${PROB_NUM}_noBstAcc
#SAVE_SP=_pl1st_ft_update_pn${PROB_NUM}_noBstAcc
#
#NLAB=3
#BEST_ACC=0
#PROB_FILE_NAME=ft_nlab3_top30_ep10_wCLS.npy
#python ./code/uda_with_probing_words_1vsall_pl1st_agnews.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP


# ------- update probing words with consistent ratio on unlabeled val set --------
GPU=0,1

PROB_WORD_TYPE=dynamic
LOSS_FUNC=bce
PROB_CONFID=0.7
PROB_TOPK=30

PROB_NUM=75
SP_NAME=_consRatio
SAVE_SP=_ft_pn${PROB_NUM}_consRatio

NLAB=10
PROB_FILE_NAME=ft_nlab10_top30_ep7_wCLS.npy
python ./code/uda_with_probing_words_1vsall_consist.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval

NLAB=5
PROB_FILE_NAME=ft_nlab5_top30_ep6_wCLS.npy
python ./code/uda_with_probing_words_1vsall_pl1st_consist.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval

NLAB=3
PROB_FILE_NAME=ft_nlab3_top30_ep10_wCLS.npy
python ./code/uda_with_probing_words_1vsall_pl1st_consist.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval