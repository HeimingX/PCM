#!/usr/bin/env bash

LBS=4
UBS=8

T=0.5
DATASET=./data/yahoo_answers_csv/yahoo_answers_csv/

# ================================================ original uda ========================================================
GPU=0,1 #,2
# --classify-with-cls --add-cls-sep --label-smooth
#python ./code/uda.py --gpu $GPU --n-labeled 10 --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _gapFeaOnly --eval --resume ./ckpt/uda/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_seed0_gapFeaOnly_ep1.ckpt

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

#python ./code/uda.py --gpu 0,1,2 --n-labeled 10 \
#	--data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _aug1st2pl

# ================================================ uda with pre-defined probing words list ================================================
#python ./code/uda_with_probing_words.py --gpu $GPU --n-labeled 10 --add-cls-sep \
#	--data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _matchClsfr_gapFeaOnly --eval #--resume ./ckpt/udaPrbWrd/checkpoint.nU5000_wClsSep_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd_matchClsfr_gapFeaOnly_ep9.ckpt

#checkpoint.nU5000_wClsSep_lr5e-06_0.0005_ep20_it1000_lbs8_ubs24_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd_matchClsfr_gapFeaOnly_ep11.ckpt
#checkpoint.nU5000_wClsSep_lr5e-06_0.0005_ep20_it1000_lbs8_ubs24_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd_matchClsfr_ep2.ckpt
#checkpoint.nU5000_noClsSep_lr5e-06_0.0005_ep20_it1000_lbs8_ubs24_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd_matchClsfr_ep3.ckpt
#checkpoint.nU5000_noClsSep_lr5e-06_0.0005_ep20_it1000_lbs8_ubs18_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd_matchClsfr_ep8.ckpt
#checkpoint.nU5000_noClsSep_lr5e-06_0.0005_ep20_it1000_lbs8_ubs18_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd_ep9.ckpt

# ================================================ dynamic weight ================================================
#GPU=0,1,2
#GPU=1,2 #,2
#GPU=1 #,2 --weighted_prob_word
#GPU=2 #,2 --weighted_prob_word
#NLAB=200
#NLAB=2500 --prob_word_num 20
NLAB=10
#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T --wProb 1.0 --confid_prob 0.7 --weighted_prob_word \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _matchClsfr_gapFeaOnly_bce_snglPrbPl_dynWeight #--resume ./ckpt/udaPrbWrd/ --eval
# _mixtext
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd1.0_cnfdPrb0.7_seed0_matchClsfr_gapFeaOnly_bce_snglPrbPl_nlab200_ep19.ckpt
#_nlab200  _allToken #_allToken  
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd1.0_cnfdPrb0.7_seed0_matchClsfr_gapFeaOnly_bce_snglPrbPl_nlab2500_ep12.ckpt
#checkpoint.nU5000_noClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd1.0_cnfdPrb0.7_seed0_matchClsfr_bce_snglPrbPl_ep16.ckpt
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd1.0_cnfdPrb0.75_seed0_matchClsfr_gapFeaOnly_bce_snglPrbPl_ep15.ckpt
#checkpoint.nU5000_noClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd1.0_cnfdPrb0.7_seed0_matchClsfr_bce_snglPrbPl_ep0.ckpt
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd1.0_cnfdPrb0.65_seed0_matchClsfr_gapFeaOnly_bce_snglPrbPl_ep1.ckpt
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd1.0_cnfdPrb0.75_seed0_matchClsfr_gapFeaOnly_bce_snglPrbPl_ep15.ckpt
#_mrgn5 _gapFeaOnly
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd1.0_cnfdPrb0.7_seed0_matchClsfr_gapFeaOnly_bce_snglPrbPl_mrgn5_ep19.ckpt
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd1.0_cnfdPrb0.7_seed0_matchClsfr_gapFeaOnly_bce_snglPrbPl_ep11.ckpt
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd1.0_cnfdPrb0.7_seed0_matchClsfr_gapFeaOnly_bce_snglPrbPl_mrgn10_ep12.ckpt _mrgn10
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd1.0_cnfdPrb0.7_matchClsfr_gapFeaOnly_bce_snglPrbPl_ep18.ckpt
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd1.0_cnfdPrb0.7_matchClsfr_gapFeaOnly_bce_snglPrbPl_rerun3_ep19.ckpt
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd1.0_cnfdPrb0.7_matchClsfr_gapFeaOnly_bce_snglPrbPl_rerun_ep14.ckpt
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd1.0_cnfdPrb0.7_matchClsfr_gapFeaOnly_bce_snglPrbPl_mrgn10_ep13.ckpt
#3test_rerun3
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd1.0_cnfdPrb0.8_matchClsfr_gapFeaOnly_bce_snglPrbPl_ep18.ckpt
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd1.0_cnfdPrb0.7_matchClsfr_gapFeaOnly_bce_snglPrbPl_ep18.ckpt
#_mrgn10
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd1.0_cnfdPrb0.6_matchClsfr_gapFeaOnly_bce_snglPrbPl_ep10.ckpt
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd1.0_matchClsfr_gapFeaOnly_bce_snglPrbPl_ep13.ckpt
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd_matchClsfr_gapFeaOnly_bce_ep2.ckpt
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd10.0_matchClsfr_gapFeaOnly_bce_ep13.ckpt
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd_matchClsfr_gapFeaOnly_bce_ep16.ckpt
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd_matchClsfr_gapFeaOnly_1vsall_ep8.ckpt
#1vsall bce#
#checkpoint.nU5000_wClsSep_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd_matchClsfr_gapFeaOnly_ep9.ckpt

#python ./code/uda_with_probing_words_split_match.py --gpu $GPU --n-labeled 10 --add-cls-sep --split-match \
#	--data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _matchClsfr

# ================================================ various prob word type ================================================
#  --dropout_prob 0
# probing words from mixtext
#sleep 2h --prob_word_num 50  uda
#GPU=0,1 #,2 
#GPU=0,2 #,2
#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--prob_word_type uda --prob_word_num 150 --confid_prob 0.7 --dropout_prob 0.0 \
#	--lambda-u 1 --T $T --wProb 1.0 \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _matchClsfr_gapFeaOnly_bce_snglPrbPl #--resume ./ckpt/udaPrbWrd/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromdynamic_NoParallel_prbNum150_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.7_seed0_matchClsfr_gapFeaOnly_bce_snglPrbPl_ep4.ckpt --eval

#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromdynamic_NoParallel_prbNum150_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.7_seed0_matchClsfr_gapFeaOnly_bce_snglPrbPl_rerun_ep5.ckpt
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromdynamic_NoParallel_prbNum0_drpPrb0.5_wPrbWrd1.0_cnfdPrb0.7_seed0_matchClsfr_gapFeaOnly_bce_snglPrbPl_ep4.ckpt

GPU=0,1
PROB_NUM=75
#PROB_NUM=50
#PROB_NUM=60
#PROB_NUM=40
#PROB_NUM=30

#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--prob_word_type uda --prob_word_num $PROB_NUM --parallel_prob --confid_prob 0.7 --dropout_prob 0.0 \
#	--lambda-u 1 --T $T --wProb 1.0 \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _matchClsfr_gapFeaOnly_bce_snglPrbPl #--resume ./ckpt/udaPrbWrd/ --eval

python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
	--prob_word_type uda --prob_word_num $PROB_NUM --parallel_prob --confid_prob 0.7 --dropout_prob 0.0 --weighted_prob_word --weighted_type softmax \
	--lambda-u 1 --T $T --wProb 1.0 \
	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _matchClsfr_gapFeaOnly_bce_snglPrbPl --resume ./experiments/uda_probing_words/yahoo_answers_10/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromuda_weighted_typesoftmax_parallel_prbNum75_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.7_seed0_matchClsfr_gapFeaOnly_bce_snglPrbPl_ep12.ckpt --eval


#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size $LBS --batch-size-u $UBS \
#	--prob_word_type dynamic --prob_word_num 100 --confid_prob 0.7 --dropout_prob 0.5 \
#	--lambda-u 1 --T $T --wProb 1.0 \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _matchClsfr_gapFeaOnly_bce_snglPrbPl #--resume ./ckpt/udaPrbWrd/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromdynamic_NoParallel_prbNum0_drpPrb0.5_wPrbWrd1.0_cnfdPrb0.7_seed0_matchClsfr_gapFeaOnly_bce_snglPrbPl_ep4.ckpt


