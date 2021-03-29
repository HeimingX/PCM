#!/usr/bin/env bash

LBS=4
UBS=8

T=0.5
DATASET=./data/yahoo_answers_csv/yahoo_answers_csv/

# ====== ablation study: verify the contributed loss term on uda =========
GPU=0,1
#NLAB=10
NLAB=5
#NLAB=4
#NLAB=3
#NLAB=2
#NLAB=1
SP=_gapFeaOnly_bce0.7_run2
python ./code/uda_bce.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
	--lambda-u 1 --T $T --confid_prob 0.7 \
	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name $SP #--eval --resume ./experiments/uda/

# ===== ablation study: verify the generation of pseudo label ========
#python ./code/uda.py --gpu 0,1,2 --n-labeled 10 \
#	--data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _aug1st2pl

# ======= uda with pre-defined probing words list =========
#python ./code/uda_with_probing_words.py --gpu $GPU --n-labeled 10 --add-cls-sep \
#	--data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _matchClsfr_gapFeaOnly --eval #--resume ./ckpt/udaPrbWrd/checkpoint.nU5000_wClsSep_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd_matchClsfr_gapFeaOnly_ep9.ckpt

#checkpoint.nU5000_wClsSep_lr5e-06_0.0005_ep20_it1000_lbs8_ubs24_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd_matchClsfr_gapFeaOnly_ep11.ckpt
#checkpoint.nU5000_wClsSep_lr5e-06_0.0005_ep20_it1000_lbs8_ubs24_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd_matchClsfr_ep2.ckpt
#checkpoint.nU5000_noClsSep_lr5e-06_0.0005_ep20_it1000_lbs8_ubs24_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd_matchClsfr_ep3.ckpt
#checkpoint.nU5000_noClsSep_lr5e-06_0.0005_ep20_it1000_lbs8_ubs18_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd_matchClsfr_ep8.ckpt
#checkpoint.nU5000_noClsSep_lr5e-06_0.0005_ep20_it1000_lbs8_ubs18_noTsa_wu1.0_T0.5_confid0.95_wPrbWrd_ep9.ckpt

# ==== split match each class probing-words =====
#python ./code/uda_with_probing_words_split_match.py --gpu $GPU --n-labeled 10 --add-cls-sep --split-match \
#	--data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _matchClsfr


# ================================================ various prob word type on two classifiers ===========================
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
GPU=1,2
#PROB_NUM=75
#PROB_NUM=50
PROB_NUM=60
#PROB_NUM=40
#PROB_NUM=45
#PROB_NUM=30
#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--prob_word_type uda --prob_word_num $PROB_NUM --parallel_prob --confid_prob 0.7 --dropout_prob 0.0 \
#	--lambda-u 1 --T $T --wProb 1.0 \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _matchClsfr_gapFeaOnly_bce_snglPrbPl_rerun #--resume ./experiments/uda_probing_words/yahoo_answers_10/ckpt/ --eval

#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromuda_parallel_prbNum75_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.7_seed0_matchClsfr_gapFeaOnly_bce_snglPrbPl_ep15.ckpt
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromuda_parallel_prbNum40_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.7_seed0_matchClsfr_gapFeaOnly_bce_snglPrbPl_ep11.ckpt
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromuda_parallel_prbNum50_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.7_seed0_matchClsfr_gapFeaOnly_bce_snglPrbPl_ep11.ckpt
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromuda_parallel_prbNum60_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.7_seed0_matchClsfr_gapFeaOnly_bce_snglPrbPl_ep10.ckpt

#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size $LBS --batch-size-u $UBS \
#	--prob_word_type dynamic --prob_word_num 100 --confid_prob 0.7 --dropout_prob 0.5 \
#	--lambda-u 1 --T $T --wProb 1.0 \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _matchClsfr_gapFeaOnly_bce_snglPrbPl #--resume ./ckpt/udaPrbWrd/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromdynamic_NoParallel_prbNum0_drpPrb0.5_wPrbWrd1.0_cnfdPrb0.7_seed0_matchClsfr_gapFeaOnly_bce_snglPrbPl_ep4.ckpt

#PROB_NUM=40
PROB_NUM=45
#PROB_NUM=75
#PROB_NUM=150
#PROB_NUM=200
#PROB_NUM=250
#PROB_NUM=300
WEIGHTED_TYPE=softmax
WEIGHTED_TYPE=reluNorm

#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--prob_word_type uda --prob_word_num $PROB_NUM --parallel_prob --confid_prob 0.7 --dropout_prob 0.0 --weighted_prob_word --weighted_type $WEIGHTED_TYPE \
#	--lambda-u 1 --T $T --wProb 1.0 \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _matchClsfr_gapFeaOnly_bce_snglPrbPl #--resume ./experiments/uda_probing_words/yahoo_answers_10/ckpt/ --eval

#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromuda_weighted_typereluNorm_parallel_prbNum40_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.7_seed0_matchClsfr_gapFeaOnly_bce_snglPrbPl_ep15.ckpt
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromuda_weighted_typereluNorm_parallel_prbNum150_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.7_seed0_matchClsfr_gapFeaOnly_bce_snglPrbPl_ep11.ckpt
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromuda_weighted_typesoftmax_parallel_prbNum75_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.7_seed0_matchClsfr_gapFeaOnly_bce_snglPrbPl_ep12.ckpt