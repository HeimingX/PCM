#!/usr/bin/env bash

LBS=4
UBS=8

T=0.5
DATASET=./data/yahoo_answers_csv/yahoo_answers_csv/

# ================================================ original uda =============================================
GPU=0 #,2
SEED=0

#NLAB=10
#NLAB=5
#NLAB=4
#NLAB=3
#NLAB=2
NLAB=1
#python ./code/uda.py --gpu $GPU --n-labeled $NLAB \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _rerun2 #--eval --resume ./experiments/uda/yahoo_answers_5/ckpt/checkpoint.nU5000_noClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_seed0_ep17.ckpt

#yahoo_answers_4/ckpt/checkpoint.nU5000_noClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_seed0_ep16.ckpt
#yahoo_answers_3/ckpt/checkpoint.nU5000_noClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs3_ubs8_noTsa_wu1.0_T0.5_confid0.95_seed0_ep8.ckpt
##yahoo_answers_2/ckpt/checkpoint.nU5000_noClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs2_ubs8_noTsa_wu1.0_T0.5_confid0.95_seed0_ep12.ckpt
#yahoo_answers_1/ckpt/checkpoint.nU5000_noClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs1_ubs8_noTsa_wu1.0_T0.5_confid0.95_seed0_ep0.ckpt

GPU=0,1
NLAB=10
TOPK=5
NLAB=5
TOPK=30
NLAB=4
NLAB=3
NLAB=2
#NLAB=1
#python ./code/uda.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T --prob_topk $TOPK \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _gapFeaOnly --eval --resume ./experiments/uda/yahoo_answers_2/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_seed0_gapFeaOnly_ep3.ckpt

#yahoo_answers_3/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_seed0_gapFeaOnly_ep12.ckpt
#yahoo_answers_4/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_seed0_gapFeaOnly_ep12.ckpt
#yahoo_answers_5/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_seed0_ep14.ckpt
#yahoo_answers_10/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_seed0_gapFeaOnly_ep4.ckpt
#yahoo_answers_3/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_seed0_gapFeaOnly_ep12.ckpt
#_rerun3

#./ckpt/uda/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_seed0_gapFeaOnly_ep1.ckpt
#yahoo_answers_3/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs3_ubs8_noTsa_wu1.0_T0.5_confid0.95_seed0_gapFeaOnly_ep4.ckpt
#yahoo_answers_2/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs2_ubs8_noTsa_wu1.0_T0.5_confid0.95_seed0_gapFeaOnly_ep4.ckpt
#yahoo_answers_1/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs1_ubs8_noTsa_wu1.0_T0.5_confid0.95_seed0_gapFeaOnly_ep8.ckpt
#yahoo_answers_4/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_seed0_gapFeaOnly_ep12.ckpt
#yahoo_answers_5/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_seed0_ep14.ckpt
#yahoo_answers_3/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_seed0_gapFeaOnly_ep12.ckpt
#yahoo_answers_2/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_seed0_gapFeaOnly_ep3.ckpt
#yahoo_answers_1/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_seed0_gapFeaOnly_ep0.ckpt

#checkpoint.nU5000_noClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_seed0_ep16.ckpt
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_seed0_gapFeaOnly_ep1.ckpt
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_gapFeaOnly_gtVerifyPl_ep19.ckpt
#_gtVerifyPl
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_gapFeaOnly_ep3.ckpt
#checkpoint.nU5000_noClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_ep2.ckpt  
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_gapFeaOnly_ep3.ckpt
#checkpoint.nU5000_wClsSep_lr5e-06_0.0005_ep20_it1000_lbs8_ubs24_noTsa_wu1.0_T0.5_confid0.95_gapFeaOnly_ep1.ckpt 
#checkpoint.nU5000_wClsSep_noGap_lr5e-06_0.0005_ep20_it1000_lbs8_ubs24_noTsa_wu1.0_T0.5_confid0.95_ep7.ckpt
#checkpoint.nU5000_wClsSep_lr5e-06_0.0005_ep20_it1000_lbs8_ubs24_noTsa_wu1.0_T0.5_confid0.95_ep4.ckpt
#checkpoint.nU5000_noClsSep_lr5e-06_0.0005_ep20_it1000_lbs8_ubs18_noTsa_wu1.0_T0.5_confid0.95_ep17.ckpt
#checkpoint.nU5000_lr5e-06_0.0005_ep20_it1000_lbs8_ubs30_noTsa_wu1.0_T0.5_confid0.95_ep7.ckpt	
#checkpoint.nU5000_lr5e-06_0.0005_ep20_it1000_lbs8_ubs24_noTsa_wu1.0_T0.5_confid0.95_ep11.ckpt

# ====================================== single classifier: matching classifier ==============================
GPU=0,1  #1
#GPU=2,0
#GPU=1,2
PROB_NUM=75
#PROB_NUM=50
#PROB_NUM=60
#PROB_NUM=40
#PROB_NUM=45
#PROB_NUM=30

PROB_CONFID=0.7

#NLAB=4
#PROB_FILE_NAME=uda_nlab4_top5_noStopWords_probWrods_ep12.npy
#NLAB=5
#PROB_FILE_NAME=uda_nlab5_top5_noStopWords_probWrods_ep14.npy
NLAB=3
PROB_FILE_NAME=uda_nlab3_top5_noStopWords_probWrods_ep12.npy
#NLAB=2
#PROB_FILE_NAME=uda_nlab2_top30_noStopWords_probWrods_ep3.npy
#PROB_FILE_NAME=uda_nlab2_top5_noStopWords_probWrods_ep3.npy
#NLAB=1
#PROB_FILE_NAME=uda_nlab1_top30_noStopWords_probWrods_ep0.npy

#python ./code/uda_with_probing_words_1vsall_1clsfr.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type uda --prob_word_num $PROB_NUM --parallel_prob --confid_prob $PROB_CONFID --dropout_prob 0.0 \
#	--lambda-u 1 --T $T --wProb 1.0 \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _onlyMatchClsfr --resume ./experiments/uda_probing_words/yahoo_answers_3/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromuda_parallel_prbNum75_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.7_seed0_onlyMatchClsfr_ep1.ckpt --eval


#yahoo_answers_4/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromuda_parallel_prbNum75_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.7_seed0_onlyMatchClsfr_ep9.ckpt
#yahoo_answers_10/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromuda_parallel_prbNum40_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.7_seed0_onlyMatchClsfr_ep13.ckpt
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromuda_parallel_prbNum75_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.7_seed0_onlyMatchClsfr_ep13.ckpt

# ============================== dynamic probing words ========================
GPU=1,0
BEST_ACC=0
PROB_FILE_NAME=cls_names
PROB_TOPK=30

SP_NAME=_onlyMatchClsfr
SAVE_SP=_new

NLAB=10
#PROB_TOPK=30
#PROB_FILE_NAME=uda_dyn_nlab10_top30.0_noStopWords_probWrods_ep2_new.npy
#RESUME=./experiments/uda_probing_words/yahoo_answers_10/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromClsNames_dynamic_parallel_prbNum75_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.7_seed0_onlyMatchClsfr_ep12.ckpt
#PROB_TOPK=5
#PROB_FILE_NAME=uda_dyn_nlab10_top5.0_noStopWords_probWrods_ep4_new1.npy
#RESUME=./experiments/uda_probing_words/yahoo_answers_10/ckpt/ph_rerun/best.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromClsNames_dynamic_parallel_prbNum75_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.7_seed0_onlyMatchClsfr.ckpt
# probing words start from uda
#PROB_FILE_NAME=uda_dyn_nlab10_top5.0_noStopWords_probWrods_ep5_new2.npy
#RESUME=./experiments/uda_probing_words/yahoo_answers_10/ckpt/best.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromUDA_dynamic_parallel_prbNum75_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.7_seed0_onlyMatchClsfr.ckpt

NLAB=5
#PROB_FILE_NAME=uda_dyn_nlab5_top30.0_noStopWords_probWrods_ep7rerun.npy
#PROB_FILE_NAME=uda_dyn_nlab5_top30.0_noStopWords_probWrods_ep4_new.npy
#RESUME=./experiments/uda_probing_words/yahoo_answers_5/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromClsNames_dynamic_parallel_prbNum75_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.7_seed0_onlyMatchClsfr_ep8.ckpt
# probing words start from uda
#PROB_FILE_NAME=uda_dyn_nlab5_top5.0_noStopWords_probWrods_ep3_new2.npy
#RESUME=./experiments/uda_probing_words/yahoo_answers_5/ckpt/best.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromUDA_dynamic_parallel_prbNum75_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.7_seed0_onlyMatchClsfr.ckpt

NLAB=4
#PROB_FILE_NAME=uda_dyn_nlab4_top30.0_noStopWords_probWrods_ep6.npy
#PROB_FILE_NAME=uda_dyn_nlab4_top30.0_noStopWords_probWrods_ep13_new.npy
#RESUME=./experiments/uda_probing_words/yahoo_answers_4/ckpt/best.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromClsNames_dynamic_parallel_prbNum75_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.7_seed0_onlyMatchClsfr.ckpt
#PROB_FILE_NAME=uda_dyn_nlab4_top30.0_noStopWords_probWrods_ep1_new1.npy
#RESUME=./experiments/uda_probing_words/yahoo_answers_4/ckpt/ph_rerun/best.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromClsNames_dynamic_parallel_prbNum75_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.7_seed0_onlyMatchClsfr.ckpt
#PROB_FILE_NAME=uda_dyn_nlab4_top30.0_noStopWords_probWrods_ep1_new2.npy
#RESUME=./experiments/uda_probing_words/yahoo_answers_4/ckpt/best.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromUDA_dynamic_parallel_prbNum75_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.7_seed0_onlyMatchClsfr.ckpt

NLAB=3
#PROB_FILE_NAME=uda_dyn_nlab3_top30.0_noStopWords_probWrods_ep3_new.npy
#PROB_FILE_NAME=uda_dyn_nlab3_top30.0_noStopWords_probWrods_ep5.npy
#RESUME=./experiments/uda_probing_words/yahoo_answers_3/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromClsNames_dynamic_parallel_prbNum75_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.7_seed0_onlyMatchClsfr_ep6.ckpt
#PROB_FILE_NAME=uda_dyn_nlab3_top30.0_noStopWords_probWrods_ep15_new2.npy
#RESUME=./experiments/uda_probing_words/yahoo_answers_3/ckpt/best.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromUDA_dynamic_parallel_prbNum75_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.7_seed0_onlyMatchClsfr.ckpt

NLAB=2
PROB_FILE_NAME=uda_dyn_nlab2_top30.0_noStopWords_probWrods_ep4_new.npy
RESUME=./experiments/uda_probing_words/yahoo_answers_2/ckpt/best.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromClsNames_dynamic_parallel_prbNum75_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.7_seed0_onlyMatchClsfr.ckpt
PROB_FILE_NAME=uda_dyn_nlab2_top30.0_noStopWords_probWrods_ep1_new2.npy
RESUME=./experiments/uda_probing_words/yahoo_answers_2/ckpt/best.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromUDA_dynamic_parallel_prbNum75_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.7_seed0_onlyMatchClsfr.ckpt
python ./code/uda_with_probing_words_1vsall_1clsfr.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
	--prob_word_type dynamic --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --init_best_acc $BEST_ACC --parallel_prob --confid_prob $PROB_CONFID --dropout_prob 0.0 \
	--lambda-u 1 --T $T --wProb 1.0 --local_machine \
	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP --resume $RESUME --eval
#
#best.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromClsNames_dynamic_parallel_prbNum75_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.7_seed0_onlyMatchClsfr.ckpt
NLAB=5
#python ./code/uda_with_probing_words_1vsall_1clsfr.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type dynamic --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --init_best_acc $BEST_ACC --parallel_prob --confid_prob $PROB_CONFID --dropout_prob 0.0 \
#	--lambda-u 1 --T $T --wProb 1.0 \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval

NLAB=4
#python ./code/uda_with_probing_words_1vsall_1clsfr.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type dynamic --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --init_best_acc $BEST_ACC --parallel_prob --confid_prob $PROB_CONFID --dropout_prob 0.0 \
#	--lambda-u 1 --T $T --wProb 1.0 \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval

NLAB=3
#python ./code/uda_with_probing_words_1vsall_1clsfr.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type dynamic --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --init_best_acc $BEST_ACC --parallel_prob --confid_prob $PROB_CONFID --dropout_prob 0.0 \
#	--lambda-u 1 --T $T --wProb 1.0 \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval

NLAB=2
#python ./code/uda_with_probing_words_1vsall_1clsfr.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type dynamic --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --init_best_acc $BEST_ACC --parallel_prob --confid_prob $PROB_CONFID --dropout_prob 0.0 \
#	--lambda-u 1 --T $T --wProb 1.0 \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval

NLAB=1
#python ./code/uda_with_probing_words_1vsall_1clsfr.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type dynamic --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --init_best_acc $BEST_ACC --parallel_prob --confid_prob $PROB_CONFID --dropout_prob 0.0 \
#	--lambda-u 1 --T $T --wProb 1.0 \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval



#yahoo_answers_3/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromdynamic_parallel_prbNum75_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.7_seed0_onlyMatchClsfr_ep11.ckpt
#yahoo_answers_4/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromdynamic_parallel_prbNum75_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.7_seed0_onlyMatchClsfr_ep17.ckpt

#sleep 3h
#BEST_ACC=0.6539
#PROB_FILE_NAME=uda_top5_noStopWords_probWrods_ep1.npy

#BEST_ACC=0.6432
#PROB_FILE_NAME=uda_nlab5_top5_noStopWords_probWrods_ep14.npy

#BEST_ACC=0.5009
#PROB_FILE_NAME=uda_nlab4_top5_noStopWords_probWrods_ep12.npy
#---eval---
#PROB_FILE_NAME=uda_dyn_nlab4_top30.0_noStopWords_probWrods_ep6.npy

#BEST_ACC=0.4221
#PROB_FILE_NAME=uda_nlab3_top30_noStopWords_probWrods_ep12.npy
# ---eval ---
#PROB_FILE_NAME=uda_dyn_nlab3_top30.0_noStopWords_probWrods_ep10.npy

#PROB_TOPK=30
#BEST_ACC=0.3355
#PROB_FILE_NAME=uda_nlab2_top30_noStopWords_probWrods_ep3.npy
#PROB_FILE_NAME=uda_nlab2_top5_noStopWords_probWrods_ep3.npy

#PROB_TOPK=30
#BEST_ACC=0.1661
#NLAB=1
#PROB_FILE_NAME=uda_nlab1_top30_noStopWords_probWrods_ep0.npy


# ============================== weighted probing words :relu + norm ========================
#WEIGHTED_TYPE=softmax
WEIGHTED_TYPE=reluNorm

#sleep 2h
GPU=0,1
#NLAB=4
#PROB_FILE_NAME=uda_nlab4_top5_noStopWords_probWrods_ep12.npy
#NLAB=5
#PROB_FILE_NAME=uda_nlab5_top5_noStopWords_probWrods_ep14.npy
#NLAB=3
#PROB_FILE_NAME=uda_nlab3_top5_noStopWords_probWrods_ep12.npy
NLAB=2
PROB_FILE_NAME=uda_nlab2_top30_noStopWords_probWrods_ep3.npy
#PROB_FILE_NAME=uda_nlab2_top5_noStopWords_probWrods_ep3.npy
#NLAB=1
#PROB_FILE_NAME=uda_nlab1_top30_noStopWords_probWrods_ep0.npy
#sleep 2h
#python ./code/uda_with_probing_words_1vsall_1clsfr.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type uda --prob_word_num $PROB_NUM --parallel_prob --confid_prob $PROB_CONFID --dropout_prob 0.0 --weighted_prob_word --weighted_type $WEIGHTED_TYPE \
#	--lambda-u 1 --T $T --wProb 1.0 \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _onlyMatchClsfr #--resume ./experiments/uda_probing_words/yahoo_answers_10/ckpt/ --eval







