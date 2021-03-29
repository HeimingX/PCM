#!/usr/bin/env bash

LBS=4
UBS=8

DATASET=./data/yahoo_answers_csv/yahoo_answers_csv/
LR_MAIN=5e-6
LR_LAST=5e-4
CONFID=0.95
T=0.5

PROB_WORD_TYPE=dynamic
PROB_NUM=75
#PROB_NUM=10
LOSS_FUNC=bce
PROB_CONFID=0.7

PROB_TOPK=30

# ------- update probing words with consistent ratio on unlabeled val set --------
GPU=0,1

PROB_WORD_TYPE=dynamic
LOSS_FUNC=bce
PROB_CONFID=0.7
PROB_TOPK=30

PROB_NUM=75
SP_NAME=_consRatio
SAVE_SP=_clsName_pn${PROB_NUM}_consRatio

#NLAB=10
#PROB_FILE_NAME=ft_nlab10_top30_ep19_wCLS.npy
#python ./code/uda_with_probing_words_1vsall_consist.py --gpu $GPU --n-labeled $NLAB --add-cls-sep --local_machin \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval


#NLAB=5
#PROB_FILE_NAME=ft_nlab5_top30_ep13_wCLS.npy
#python ./code/uda_with_probing_words_1vsall_consist.py --gpu $GPU --n-labeled $NLAB --add-cls-sep --local_machin \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval

PROB_NUM=5
SP_NAME=_consRatio_run2
SAVE_SP=_clsName_pn${PROB_NUM}_consRatio_run2
NLAB=3
PROB_FILE_NAME=ft_nlab3_top30_ep19_wCLS.npy
python ./code/uda_with_probing_words_1vsall_consist.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval


# ------- update probing words with fixed epoch length --------
GPU=1,2

FIX_EP=2

PROB_NUM=75
SP_NAME=_fix${FIX_EP}
SAVE_SP=_clsName_pn${PROB_NUM}_fix${FIX_EP}

NLAB=10
PROB_FILE_NAME=ft_nlab10_top30_ep19_wCLS.npy
#python ./code/uda_with_probing_words_1vsall_fixEpoch.py --gpu $GPU --n-labeled $NLAB --add-cls-sep --local_machin \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC --fix_ep $FIX_EP \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval
#
#
#NLAB=5
#PROB_FILE_NAME=ft_nlab5_top30_ep13_wCLS.npy
#python ./code/uda_with_probing_words_1vsall_fixEpoch.py --gpu $GPU --n-labeled $NLAB --add-cls-sep --local_machin \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC --fix_ep $FIX_EP \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval
#
#PROB_NUM=5
#SP_NAME=_fix${FIX_EP}
#SAVE_SP=_clsName_pn${PROB_NUM}_fix${FIX_EP}
#NLAB=3
#PROB_FILE_NAME=ft_nlab3_top30_ep19_wCLS.npy
#python ./code/uda_with_probing_words_1vsall_fixEpoch.py --gpu $GPU --n-labeled $NLAB --add-cls-sep --local_machin \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC --fix_ep $FIX_EP \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval


# ------- update probing words with quantification on unlabeled validation data --------
GPU=0,1
BEST_ACC=0
#NLAB=10
#PROB_FILE_NAME=ft_nlab10_top30_ep19_wCLS.npy

#NLAB=5
#PROB_FILE_NAME=ft_nlab5_top30_ep13_wCLS.npy

NLAB=3
PROB_NUM=5
PROB_FILE_NAME=ft_nlab3_top30_ep19_wCLS.npy

SP_NAME=_1infer_ft_quant
SAVE_SP=_1infer_ft_update_pn${PROB_NUM}_quant
#python ./code/uda_with_probing_words_1vsall_quant.py --gpu $GPU --n-labeled $NLAB --add-cls-sep --local_machin \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval


# ------ update probing words with labeled validation data -----------
#SP_NAME=_1infer_ft_wBstAcc
#SAVE_SP=_1infer_ft_update_pn75_wBstAcc
#NLAB=10
#BEST_ACC=0.6338
#PROB_FILE_NAME=ft_nlab10_top30_ep19_wCLS.npy
#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval

#NLAB=5
#BEST_ACC=0.4665
#PROB_FILE_NAME=ft_nlab5_top30_ep13_wCLS.npy
#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval

GPU=1

NLAB=3
PROB_FILE_NAME=ft_nlab3_top30_ep19_wCLS.npy
#BEST_ACC=0.4146
BEST_ACC=0


#PROB_NUM=5
#PROB_NUM=4
#PROB_NUM_UPDATE=0 # update the prob num after the acc reach the initial best eval acc
#SP_NAME=_1infer_ft_noBstAcc_pn${PROB_NUM}
#SAVE_SP=_1infer_ft_update_pn${PROB_NUM}_noBstAcc
#
#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_word_num_update $PROB_NUM_UPDATE --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval


#PROB_NUM=3
#PROB_NUM_UPDATE=0 # update the prob num after the acc reach the initial best eval acc
#SP_NAME=_1infer_ft_noBstAcc_pn${PROB_NUM}
#SAVE_SP=_1infer_ft_update_pn${PROB_NUM}_noBstAcc
#
#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_word_num_update $PROB_NUM_UPDATE --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval
#
#PROB_NUM=2
#PROB_NUM_UPDATE=0 # update the prob num after the acc reach the initial best eval acc
#SP_NAME=_1infer_ft_noBstAcc_pn${PROB_NUM}
#SAVE_SP=_1infer_ft_update_pn${PROB_NUM}_noBstAcc
#
#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_word_num_update $PROB_NUM_UPDATE --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval
#
#PROB_NUM=1
#PROB_NUM_UPDATE=0 # update the prob num after the acc reach the initial best eval acc
#SP_NAME=_1infer_ft_noBstAcc_pn${PROB_NUM}
#SAVE_SP=_1infer_ft_update_pn${PROB_NUM}_noBstAcc
#
#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_word_num_update $PROB_NUM_UPDATE --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval

#PROB_NUM=15
#PROB_NUM_UPDATE=0 # update the prob num after the acc reach the initial best eval acc
#SP_NAME=_1infer_ft_noBstAcc_pn${PROB_NUM}
#SAVE_SP=_1infer_ft_update_pn${PROB_NUM}_noBstAcc
#
#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_word_num_update $PROB_NUM_UPDATE --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval


#PROB_NUM=20
#PROB_NUM_UPDATE=0 # update the prob num after the acc reach the initial best eval acc
#SP_NAME=_1infer_ft_noBstAcc_pn${PROB_NUM}
#SAVE_SP=_1infer_ft_update_pn${PROB_NUM}_noBstAcc
#
#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_word_num_update $PROB_NUM_UPDATE --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval

#PROB_NUM=25
#PROB_NUM_UPDATE=0 # update the prob num after the acc reach the initial best eval acc
#SP_NAME=_1infer_ft_noBstAcc_pn${PROB_NUM}
#SAVE_SP=_1infer_ft_update_pn${PROB_NUM}_noBstAcc
#
#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_word_num_update $PROB_NUM_UPDATE --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval



# ----- eval ------
GPU=0,1

PROB_TOPK=30
BEST_ACC=0

#SP_NAME=_1infer_ft_wBstAcc
#SAVE_SP=_1infer_ft_update_pn75_wBstAcc

#NLAB=10
#PROB_FILE_NAME=uda_dyn_nlab10_top30_noStopWords_probWrods_ep11_1infer_ft_update_pn75_wBstAcc.npy
#RESUME=./experiments/uda_probing_words/yahoo_answers_10/ckpt/best.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromFt_dynamic_parallel_prbNum75_wPrbWrd1.0_prbLossbce_cnfdPrb0.7_seed0_1infer_ft_wBstAcc.ckpt

#NLAB=5
#PROB_FILE_NAME=uda_dyn_nlab5_top30_noStopWords_probWrods_ep8_1infer_ft_update_pn75_wBstAcc.npy
#RESUME=./experiments/uda_probing_words/yahoo_answers_${NLAB}/ckpt/best.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromFt_dynamic_parallel_prbNum75_wPrbWrd1.0_prbLossbce_cnfdPrb0.7_seed0_1infer_ft_wBstAcc.ckpt


#SP_NAME=_1infer_ft
#SAVE_SP=_1infer_ft_update_pn10

PROB_NUM=5
SP_NAME=_1infer_ft_noBstAcc_pn${PROB_NUM}
SAVE_SP=_1infer_ft_update_pn${PROB_NUM}_noBstAcc

NLAB=3
#BEST_ACC=0.4146
BEST_ACC=0
PROB_FILE_NAME=uda_dyn_nlab3_top30_noStopWords_probWrods_ep12_1infer_ft_update_pn5_noBstAcc.npy
RESUME=./experiments/uda_probing_words/yahoo_answers_${NLAB}/ckpt/best.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromFt_dynamic_parallel_prbNum5_wPrbWrd1.0_prbLossbce_cnfdPrb0.7_seed0_1infer_ft_noBstAcc_pn5.ckpt

#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --init_best_acc $BEST_ACC \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP --resume $RESUME --eval
