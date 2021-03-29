#!/usr/bin/env bash

LBS=4
UBS=8

T=1
DATASET=./data/imdb_csv/

# ============================== uda method ==============================
GPU=2

LR_MAIN=5e-6
LR_LAST=5e-4
CONFID=0.95

SAVE_SP=_pl1st
SP_NAME=_pl1st

SEED_L=2

NLAB=10
#PROB_TOPK=5
#python ./code/uda_pl1st.py --gpu $GPU --n-labeled $NLAB --add-cls-sep --seed_l $SEED_L \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T --confid $CONFID \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--prob_topk $PROB_TOPK --resume ./experiments/uda/imdb_10/ckpt/ --eval
#
##PROB_TOPK=30
#NLAB=5
#python ./code/uda_pl1st.py --gpu $GPU --n-labeled $NLAB --add-cls-sep --seed_l $SEED_L \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T --confid $CONFID \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--prob_topk $PROB_TOPK --resume ./experiments/uda/imdb_5/ckpt/ --eval
#
#NLAB=4
#python ./code/uda_pl1st.py --gpu $GPU --n-labeled $NLAB --add-cls-sep --seed_l $SEED_L \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T --confid $CONFID \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--prob_topk $PROB_TOPK --resume ./experiments/uda/imdb_4/ckpt/ --eval
#
#NLAB=3
#python ./code/uda_pl1st.py --gpu $GPU --n-labeled $NLAB --add-cls-sep --seed_l $SEED_L \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T --confid $CONFID \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--prob_topk $PROB_TOPK --resume ./experiments/uda/imdb_3/ckpt/ --eval
#
NLAB=2
#python ./code/uda_pl1st.py --gpu $GPU --n-labeled $NLAB --add-cls-sep --seed_l $SEED_L \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T --confid $CONFID \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--prob_topk $PROB_TOPK --resume ./experiments/uda/imdb_2/ckpt/ --eval


# ============================== uda probing words: start from class names ==============================
PROB_FILE_NAME=cls_names
PROB_WORD_TYPE=dynamic
PROB_NUM=30

LOSS_FUNC=bce
PROB_CONFID=0.7

GPU=3
SP_NAME=_1infer_pl1st
SAVE_SP=_1infer_pl1st_clsName_seed2

NLAB=10
PROB_TOPK=5
#python ./code/uda_with_probing_words_1vsall_pl1st.py --gpu $GPU --n-labeled $NLAB --add-cls-sep --seed_l $SEED_L \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP

NLAB=5
PROB_TOPK=30
#python ./code/uda_with_probing_words_1vsall_pl1st.py --gpu $GPU --n-labeled $NLAB --add-cls-sep --seed_l $SEED_L \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP

GPU=0,1
NLAB=4
#python ./code/uda_with_probing_words_1vsall_pl1st.py --gpu $GPU --n-labeled $NLAB --add-cls-sep --seed_l $SEED_L \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP

NLAB=3
#python ./code/uda_with_probing_words_1vsall_pl1st.py --gpu $GPU --n-labeled $NLAB --add-cls-sep --seed_l $SEED_L \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP

NLAB=2
python ./code/uda_with_probing_words_1vsall_pl1st.py --gpu $GPU --n-labeled $NLAB --add-cls-sep --seed_l $SEED_L \
	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
	--lambda-u 1 --T $T --confid $CONFID --prob_loss_func $LOSS_FUNC \
	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP

GPU=0,2
SP_NAME=_2infer_pl1st
SAVE_SP=_2infer_pl1st_clsName_seed2
#python ./code/uda_with_probing_words_1vsall_2infer_pl1st.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP

# ------ eval ----------------
#GPU=0,1
#NLAB=10
#PROB_FILE_NAME_EVAL=uda_dyn_nlab10_top5_noStopWords_probWrods_ep18_1infer_pl1st.npy
#RESUME=./experiments/uda_probing_words/imdb_10/ckpt/best.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T1.0_confid0.95_prbFromClsNames_dynamic_parallel_prbNum30_wPrbWrd1.0_prbLossbce_cnfdPrb0.7_seed0_1infer_pl1st.ckpt
#
#NLAB=5
#PROB_FILE_NAME_EVAL=uda_dyn_nlab5_top30_noStopWords_probWrods_ep18_1infer_pl1st.npy
#RESUME=./experiments/uda_probing_words/imdb_5/ckpt/best.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T1.0_confid0.95_prbFromClsNames_dynamic_parallel_prbNum30_wPrbWrd1.0_prbLossbce_cnfdPrb0.7_seed0_1infer_pl1st.ckpt
#
#NLAB=4
#PROB_FILE_NAME_EVAL=uda_dyn_nlab4_top30_noStopWords_probWrods_ep14_1infer_pl1st_clsName.npy
#RESUME=./experiments/uda_probing_words/imdb_4/ckpt/best.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T1.0_confid0.95_prbFromClsNames_dynamic_parallel_prbNum30_wPrbWrd1.0_prbLossbce_cnfdPrb0.7_seed0_1infer_pl1st_clsName.ckpt
#
#NLAB=3
#PROB_FILE_NAME_EVAL=uda_dyn_nlab3_top30_noStopWords_probWrods_ep16_1infer_pl1st.npy
#RESUME=./experiments/uda_probing_words/imdb_3/ckpt/best.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T1.0_confid0.95_prbFromClsNames_dynamic_parallel_prbNum30_wPrbWrd1.0_prbLossbce_cnfdPrb0.7_seed0_1infer_pl1st_clsName.ckpt
#
#NLAB=2
#PROB_FILE_NAME_EVAL=
#RESUME=./experiments/uda_probing_words/imdb_2/ckpt/best.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T1.0_confid0.95_prbFromClsNames_dynamic_parallel_prbNum30_wPrbWrd1.0_prbLossbce_cnfdPrb0.7_seed0_1infer_pl1st_clsName.ckpt

SP_NAME=_1infer_pl1st
SAVE_SP=_1infer_pl1st_clsName_seed1
#python ./code/uda_with_probing_words_1vsall_pl1st.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --prob_file_name_eval $PROB_FILE_NAME_EVAL \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP --resume $RESUME --eval

# ============================== uda probing words: start from UDA ========================
NLAB=10
BEST_ACC=0.895
PROB_TOPK=5
PROB_NUM=30
PROB_FILE_NAME=uda_nlab10_top5_noStopWords_probWrods_ep16_pl1st.npy
PROB_WORD_TYPE=dynamic
LOSS_FUNC=bce
PROB_CONFID=0.7


NLAB=5
BEST_ACC=0.7395
PROB_TOPK=30
PROB_NUM=30
PROB_FILE_NAME=uda_nlab5_top30_noStopWords_probWrods_ep18_pl1st.npy
PROB_WORD_TYPE=dynamic
LOSS_FUNC=bce
PROB_CONFID=0.7

NLAB=4
BEST_ACC=0.6635
PROB_FILE_NAME=uda_nlab4_top30_noStopWords_probWrods_ep19_pl1st.npy

NLAB=3
BEST_ACC=0.608
PROB_FILE_NAME=uda_nlab3_top30_noStopWords_probWrods_ep15_pl1st.npy

NLAB=2
BEST_ACC=0.7102
PROB_FILE_NAME=uda_nlab2_top30_noStopWords_probWrods_ep16_pl1st.npy

GPU=0,1
SAVE_SP=_1infer_pl1st
SP_NAME=_1infer_pl1st
#python ./code/uda_with_probing_words_1vsall_pl1st.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --init_best_acc $BEST_ACC --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP

GPU=0
SAVE_SP=_2infer_pl1st #_run2
SP_NAME=_2infer_pl1st #_run2
#python ./code/uda_with_probing_words_1vsall_2infer_pl1st.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --init_best_acc $BEST_ACC --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP

# ----- not good ----
SAVE_SP=_1clsfr_pl1st
SP_NAME=_1clsfr_pl1st
#python ./code/uda_with_probing_words_1vsall_1clsfr_pl1st.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --init_best_acc $BEST_ACC --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/imdb_10/ckpt/ #--eval


# ============================== eval ==============================

#NLAB=10
#BEST_ACC=0.895
#PROB_TOPK=5
#PROB_NUM=30
##PROB_FILE_NAME=uda_nlab10_top5_noStopWords_probWrods_ep16_pl1st.npy
#PROB_WORD_TYPE=dynamic
#LOSS_FUNC=bce
#PROB_CONFID=0.7
## eval
#PROB_FILE_NAME=uda_dyn_nlab10_top5_noStopWords_probWrods_ep15_1infer_pl1st.npy
#PROB_FILE_NAME=uda_dyn_nlab10_top5_noStopWords_probWrods_ep17_2infer_pl1st_run2.npy


NLAB=5
BEST_ACC=0.7395
PROB_TOPK=30
PROB_NUM=30
PROB_FILE_NAME=uda_nlab5_top30_noStopWords_probWrods_ep18_pl1st.npy
PROB_WORD_TYPE=dynamic
LOSS_FUNC=bce
PROB_CONFID=0.7
PROB_FILE_NAME=uda_dyn_nlab5_top30_noStopWords_probWrods_ep16_1infer_pl1st.npy


#NLAB=4
#BEST_ACC=0.6635
#PROB_FILE_NAME=uda_nlab4_top30_noStopWords_probWrods_ep19_pl1st.npy

GPU=0,1
SAVE_SP=_2infer_pl1st_run2
SP_NAME=_2infer_pl1st_run2
#python ./code/uda_with_probing_words_1vsall_2infer_pl1st.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --init_best_acc $BEST_ACC --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP --resume ./experiments/uda_probing_words/imdb_10/ckpt/best.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T1.0_confid0.95_prbFromUDA_dynamic_parallel_prbNum30_wPrbWrd1.0_prbLossbce_cnfdPrb0.7_seed0_2infer_pl1st_run2.ckpt --eval


#imdb_10/ckpt/best.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T1.0_confid0.95_prbFromUDA_dynamic_parallel_prbNum30_wPrbWrd1.0_prbLossbce_cnfdPrb0.7_seed0_2infer_pl1st_run2.ckpt

GPU=1,2
SAVE_SP=_1infer_pl1st
SP_NAME=_1infer_pl1st
#python ./code/uda_with_probing_words_1vsall_pl1st.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --init_best_acc $BEST_ACC --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP --resume ./experiments/uda_probing_words/imdb_5/ckpt/best.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T1.0_confid0.95_prbFromUDA_dynamic_parallel_prbNum30_wPrbWrd1.0_prbLossbce_cnfdPrb0.7_seed0_1infer_pl1st.ckpt --eval



#imdb_10/ckpt/best.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T1.0_confid0.95_prbFromUDA_dynamic_parallel_prbNum30_wPrbWrd1.0_prbLossbce_cnfdPrb0.7_seed0_1infer_pl1st.ckpt




# ================================================ original uda =============================================
GPU=0 #,2
SEED=0

CONFID=0.95

GPU=0,1 #2 #0
SP_NAME=_gapFeaOnly_2gpu #_local_1gpu #_rerun3_again #2 #2
NLAB=10
LR_MAIN=5e-6
#LR_MAIN=5e-4

#LR_LAST=5e-6
#LR_LAST=5e-4
LR_LAST=5e-3
#LR_LAST=5e-2
#python ./code/uda.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T --confid $CONFID \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME #--eval --resume ./experiments/uda/imdb_10/ckpt/best.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T1.0_confid0.95_seed0_gapFeaOnly.ckpt

NLAB=5
#python ./code/uda.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T --confid $CONFID \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _gapFeaOnly #--eval --resume ./experiments/uda/

NLAB=4
#CONFID=1
#python ./code/uda.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T --confid $CONFID \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _gapFeaOnly #--eval --resume ./experiments/uda/

NLAB=3
#python ./code/uda.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T --confid $CONFID \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _gapFeaOnly #--eval --resume ./experiments/uda/

NLAB=2
#python ./code/uda.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T --confid $CONFID \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _gapFeaOnly #--eval --resume ./experiments/uda/

NLAB=1
#python ./code/uda.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T --confid $CONFID \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _gapFeaOnly #--eval --resume ./experiments/uda/

