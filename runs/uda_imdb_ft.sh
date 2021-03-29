#!/usr/bin/env bash

LBS=4
UBS=8

T=1
DATASET=./data/imdb_csv/

LR_MAIN=5e-6
LR_LAST=5e-4
CONFID=0.95

# ------- update probing words every epoch --------
#GPU=0,1
#PROB_WORD_TYPE=dynamic
#PROB_NUM=30
##PROB_NUM=10
#LOSS_FUNC=bce
#PROB_CONFID=0.7
#PROB_TOPK=30
#
#FIX_EP=2
#
#SP_NAME=_1infer_ft_fix${FIX_EP}
#SAVE_SP=_1infer_ft_update_pn${PROB_NUM}_fix${FIX_EP}
#
#BEST_ACC=0
#
#NLAB=10
#PROB_FILE_NAME=ft_nlab10_top30_ep18_wCLS.npy
#python ./code/uda_with_probing_words_1vsall_pl1st_fixEpoch.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC --fix_ep $FIX_EP \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval
#
#NLAB=5
#PROB_FILE_NAME=ft_nlab5_top30_ep14_wCLS.npy
#python ./code/uda_with_probing_words_1vsall_pl1st_fixEpoch.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC --fix_ep $FIX_EP \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval
#
#NLAB=3
#PROB_FILE_NAME=ft_nlab3_top30_ep19_wCLS.npy
#python ./code/uda_with_probing_words_1vsall_pl1st_fixEpoch.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC --fix_ep $FIX_EP \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval

# ------- update probing words with quantification on unlabeled validation data --------
#GPU=0,1
#PROB_WORD_TYPE=dynamic
#PROB_NUM=30
##PROB_NUM=10
#LOSS_FUNC=bce
#PROB_CONFID=0.7
#PROB_TOPK=30
#SP_NAME=_1infer_ft_quant_wConfid
#SAVE_SP=_1infer_ft_update_pn${PROB_NUM}_quant_wConfid
#
#BEST_ACC=0
#
#NLAB=10
#PROB_FILE_NAME=ft_nlab10_top30_ep18_wCLS.npy
#python ./code/uda_with_probing_words_1vsall_pl1st_quant.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval
#
#NLAB=5
#PROB_FILE_NAME=ft_nlab5_top30_ep14_wCLS.npy
#python ./code/uda_with_probing_words_1vsall_pl1st_quant.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval
#
#NLAB=3
#PROB_FILE_NAME=ft_nlab3_top30_ep19_wCLS.npy
#python ./code/uda_with_probing_words_1vsall_pl1st_quant.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval


# ------- update probing words with quantification on unlabeled validation data --------
GPU=0,1
PROB_WORD_TYPE=dynamic
LOSS_FUNC=bce
PROB_TOPK=30

#PROB_NUM=30
PROB_NUM=20
#PROB_NUM=10

#PROB_CONFID=0.7
#PROB_CONFID=0.65
#PROB_CONFID=0.75
#PROB_CONFID=0.8
PROB_CONFID=0.9
#PROB_CONFID=0.91
SP_NAME=_ft_consRatio
SAVE_SP=_ft_update_pn${PROB_NUM}_consRatio_pc${PROB_CONFID}

BEST_ACC=0

#NLAB=10
#PROB_FILE_NAME=ft_nlab10_top30_ep18_wCLS.npy
#python ./code/uda_with_probing_words_1vsall_pl1st_consist.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval

NLAB=5
PROB_FILE_NAME=ft_nlab5_top30_ep14_wCLS.npy
python ./code/uda_with_probing_words_1vsall_pl1st_consist.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval

PROB_NUM=15
PROB_CONFID=0.9
SP_NAME=_ft_consRatio
SAVE_SP=_ft_update_pn${PROB_NUM}_consRatio_pc${PROB_CONFID}
python ./code/uda_with_probing_words_1vsall_pl1st_consist.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval

PROB_NUM=10
PROB_CONFID=0.9
SP_NAME=_ft_consRatio
SAVE_SP=_ft_update_pn${PROB_NUM}_consRatio_pc${PROB_CONFID}
python ./code/uda_with_probing_words_1vsall_pl1st_consist.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval

PROB_NUM=5
PROB_CONFID=0.9
SP_NAME=_ft_consRatio
SAVE_SP=_ft_update_pn${PROB_NUM}_consRatio_pc${PROB_CONFID}
python ./code/uda_with_probing_words_1vsall_pl1st_consist.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval

#NLAB=3
#PROB_FILE_NAME=ft_nlab3_top30_ep19_wCLS.npy
#python ./code/uda_with_probing_words_1vsall_pl1st_consist.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval

# ------- update probing words with quantification on unlabeled validation data --------
GPU=0,1
PROB_WORD_TYPE=dynamic
PROB_NUM=30
#PROB_NUM=10
LOSS_FUNC=bce
PROB_CONFID=0.7
PROB_TOPK=30
SP_NAME=_1infer_ft_consRatio_noConfid
SAVE_SP=_1infer_ft_update_pn${PROB_NUM}_consRatio_noConfid

BEST_ACC=0

#NLAB=10
#PROB_FILE_NAME=ft_nlab10_top30_ep18_wCLS.npy
#python ./code/uda_with_probing_words_1vsall_pl1st_consist_noConfid.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval

#NLAB=5
#PROB_FILE_NAME=ft_nlab5_top30_ep14_wCLS.npy
#python ./code/uda_with_probing_words_1vsall_pl1st_consist_noConfid.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval
#
#NLAB=3
#PROB_FILE_NAME=ft_nlab3_top30_ep19_wCLS.npy
#python ./code/uda_with_probing_words_1vsall_pl1st_consist_noConfid.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval

# ============================== uda probing words: start from class names ==============================
PROB_WORD_TYPE=dynamic
PROB_NUM=30
#PROB_NUM=10
LOSS_FUNC=bce
PROB_CONFID=0.7

SP_NAME=_1infer_ft_update_wBstAcc
SAVE_SP=_1infer_ft_update_pn30_wBstAcc

GPU=3

NLAB=10
PROB_TOPK=30
BEST_ACC=0.8950
PROB_FILE_NAME=ft_nlab10_top30_ep18_wCLS.npy
#python ./code/uda_with_probing_words_1vsall_pl1st.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP

NLAB=5
BEST_ACC=0.7395
PROB_FILE_NAME=ft_nlab5_top30_ep14_wCLS.npy
#python ./code/uda_with_probing_words_1vsall_pl1st.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP

#NLAB=4
#python ./code/uda_with_probing_words_1vsall_pl1st.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP

NLAB=3
BEST_ACC=0.608
PROB_FILE_NAME=ft_nlab3_top30_ep19_wCLS.npy
#python ./code/uda_with_probing_words_1vsall_pl1st.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP

NLAB=2
#python ./code/uda_with_probing_words_1vsall_pl1st.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP
