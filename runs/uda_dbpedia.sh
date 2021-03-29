#!/usr/bin/env bash

LBS=4
UBS=8

DATASET=./data/dbpedia_csv/
T=0.5
# ================ original uda ========================
GPU=2
LR_MAIN=5e-6
LR_LAST=5e-4
CONFID=0.95

NLAB=10
#python ./code/uda.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _gapFeaOnly #--eval --resume ./experiments/uda/
NLAB=5
#python ./code/uda.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _gapFeaOnly #--eval --resume ./experiments/uda/
NLAB=4
#python ./code/uda.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _gapFeaOnly #--eval --resume ./experiments/uda/
NLAB=3
#python ./code/uda.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _gapFeaOnly #--eval --resume ./experiments/uda/
NLAB=2
#python ./code/uda.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _gapFeaOnly #--eval --resume ./experiments/uda/
NLAB=1
#python ./code/uda.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _gapFeaOnly #--eval --resume ./experiments/uda/


# ------- update probing words with consistent ratio on unlabeled val set --------
GPU=0,1

PROB_FILE_NAME=cls_names
PROB_WORD_TYPE=dynamic
LOSS_FUNC=bce
#PROB_CONFID=0.7
PROB_CONFID=0.8
PROB_TOPK=30

PROB_NUM=75
#PROB_NUM=50
SP_NAME=_consRatio
SAVE_SP=_clsName_pn${PROB_NUM}_consRatio_pc${PROB_CONFID}

#NLAB=10
#python ./code/uda_with_probing_words_1vsall_consist.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval
#
#NLAB=5
#python ./code/uda_with_probing_words_1vsall_consist.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval

PROB_CONFID=0.7
PROB_NUM=5
#PROB_NUM=50
SP_NAME=_consRatio
SAVE_SP=_clsName_pn${PROB_NUM}_consRatio_pc${PROB_CONFID}

NLAB=3
python ./code/uda_with_probing_words_1vsall_consist.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval

PROB_CONFID=0.8
PROB_NUM=5
#PROB_NUM=50
SP_NAME=_consRatio
SAVE_SP=_clsName_pn${PROB_NUM}_consRatio_pc${PROB_CONFID}

NLAB=3
python ./code/uda_with_probing_words_1vsall_consist.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval


# ============================== dynamic probing words: start from class names ========================
PROB_FILE_NAME=cls_names
PROB_WORD_TYPE=dynamic
PROB_NUM=75
LOSS_FUNC=bce
PROB_CONFID=0.7

SP_NAME=_1infer_update
SAVE_SP=_1infer_clsName_update

GPU=0,2
NLAB=10
PROB_TOPK=5
#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval
#
##GPU=0
#NLAB=5
#PROB_TOPK=30
##SP_NAME=_1infer_run2
##SAVE_SP=_1infer_clsName_run2
#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval
#
#NLAB=4
#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval
#
#NLAB=3
#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval
#
#NLAB=2
##PROB_CONFID=0.8
##SP_NAME=_1infer_run2
##SAVE_SP=_1infer_clsName_run2
#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval


# ---------- eval -------------
#PROB_FILE_NAME_EVAL=
#RESUME=
#SP_NAME=_1infer
#SAVE_SP=_1infer
#python ./code/uda_with_probing_words_1vsall_pl1st.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --prob_file_name_eval $PROB_FILE_NAME_EVAL \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP --resume $RESUME --eval



# ============================== dynamic probing words: start from UDA ========================
GPU=0,1

NLAB=10
BEST_ACC=
SP_NAME=_onlyMatchClsfr
SAVE_SP=_new
#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --init_best_acc $BEST_ACC --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --prob_loss_func $LOSS_FUNC \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval



# ============= single classifier =============
#NLAB=10
#python ./code/uda_with_probing_words_1vsall_1clsfr.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type dynamic --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --init_best_acc $BEST_ACC --parallel_prob --confid_prob $PROB_CONFID --dropout_prob 0.0 \
#	--lambda-u 1 --T $T --wProb 1.0 \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval
#
#NLAB=5
#python ./code/uda_with_probing_words_1vsall_1clsfr.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type dynamic --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --init_best_acc $BEST_ACC --parallel_prob --confid_prob $PROB_CONFID --dropout_prob 0.0 \
#	--lambda-u 1 --T $T --wProb 1.0 \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval
#
#NLAB=4
#python ./code/uda_with_probing_words_1vsall_1clsfr.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type dynamic --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --init_best_acc $BEST_ACC --parallel_prob --confid_prob $PROB_CONFID --dropout_prob 0.0 \
#	--lambda-u 1 --T $T --wProb 1.0 \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval
#
#NLAB=3
#python ./code/uda_with_probing_words_1vsall_1clsfr.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type dynamic --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --init_best_acc $BEST_ACC --parallel_prob --confid_prob $PROB_CONFID --dropout_prob 0.0 \
#	--lambda-u 1 --T $T --wProb 1.0 \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval
#
#NLAB=2
#python ./code/uda_with_probing_words_1vsall_1clsfr.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type dynamic --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --init_best_acc $BEST_ACC --parallel_prob --confid_prob $PROB_CONFID --dropout_prob 0.0 \
#	--lambda-u 1 --T $T --wProb 1.0 \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval
#


# ================= uda with pre-defined probing words list ================
#GPU=0,1 #,2
#NLAB=10
#CONFID_PROB=0.8 #7 #8 #9
#PROB_FILE_NAME=
#PROB_FILE_NAME=
#PROB_NUM=0 #150
#PROB_NUM=150
#LRLAST=0.0005
#LRLAST=0.000005


#CONFID_PROB=0.7 #7 #8 #9
#PROB_FILE_NAME=uda_top5_noStopWords_probWrods_ep0.npy
#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--prob_word_type uda --prob_file_name $PROB_FILE_NAME --confid_prob $CONFID_PROB --dropout_prob 0.0 --prob_word_num $PROB_NUM \
#	--lambda-u 1 --T $T --wProb 1.0 \
#	--lrmain 0.000005 --lrlast $LRLAST --epochs 20 --val-iteration 1000 --specific_name _matchClsfr_gapFeaOnly_bce_snglPrbPl_prbFromUdaNoClsSep --resume ./experiments/uda_probing_words/ag_news_10/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromuda_NoParallel_prbNum150_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.7_seed0_matchClsfr_gapFeaOnly_bce_snglPrbPl_ep5.ckpt --eval


#PROB_NUM=100
#PROB_NUM=200
#PROB_NUM=150
#PROB_NUM=50
#PROB_NUM=0
#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--prob_word_type uda --prob_file_name $PROB_FILE_NAME --confid_prob $CONFID_PROB --dropout_prob 0.0 --prob_word_num $PROB_NUM \
#	--lambda-u 1 --T $T --wProb 1.0 \
#	--lrmain 0.000005 --lrlast $LRLAST --epochs 20 --val-iteration 1000 --specific_name _matchClsfr_gapFeaOnly_bce_snglPrbPl_prbFromUdaNoClsSep --resume ./experiments/uda_probing_words/ag_news_10/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromuda_NoParallel_prbNum100_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.8_seed0_matchClsfr_gapFeaOnly_bce_snglPrbPl_prbFromUdaNoClsSep_ep8.ckpt --eval