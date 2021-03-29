#!/usr/bin/env bash

LBS=4 #8 #16 #32
UBS=8 #24 #30 #24 #48 #96

T=0.5
#T=1
# ================ original uda ========================
#GPU=0,2 #,2
GPU=0
GPU=2

DATASET=./data/ag_news_csv/
# --classify-with-cls --add-cls-sep --label-smooth
#python ./code/uda.py --gpu $GPU --n-labeled 10 --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _gapFeaOnly --eval --resume ./experiments/uda/ag_news_10/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_seed0_gapFeaOnly_ep0.ckpt
#_rerun
GPU=2 #,2
#GPU=0,1 #,2
#python ./code/uda.py --gpu $GPU --n-labeled 10 \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --eval --resume ./experiments/uda/ag_news_10/ckpt/checkpoint.nU5000_noClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_seed0_ep4.ckpt

GPU=1 #,2
#GPU=0,1 #,2
#python ./code/uda.py --gpu $GPU --n-labeled 10 --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _gapFeaOnly_gtVerifyPl --eval --resume ./experiments/uda/ag_news_10/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_seed0_gapFeaOnly_gtVerifyPl_ep18.ckpt

# ================= uda with pre-defined probing words list ================
GPU=1,2 #,2
#GPU=0,2 #,2
GPU=0,1 #,2
NLAB=10
CONFID_PROB=0.8 #7 #8 #9
PROB_FILE_NAME=uda_top5_noStopWords_probWrods_ep0.npy
PROB_FILE_NAME=udaNoClsSep_top5_noStopWords_probWrods_ep4.npy
PROB_NUM=0 #150
PROB_NUM=150
LRLAST=0.0005
#LRLAST=0.000005


#CONFID_PROB=0.7 #7 #8 #9
#PROB_FILE_NAME=uda_top5_noStopWords_probWrods_ep0.npy
#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--prob_word_type uda --prob_file_name $PROB_FILE_NAME --confid_prob $CONFID_PROB --dropout_prob 0.0 --prob_word_num $PROB_NUM \
#	--lambda-u 1 --T $T --wProb 1.0 \
#	--lrmain 0.000005 --lrlast $LRLAST --epochs 20 --val-iteration 1000 --specific_name _matchClsfr_gapFeaOnly_bce_snglPrbPl_prbFromUdaNoClsSep --resume ./experiments/uda_probing_words/ag_news_10/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromuda_NoParallel_prbNum150_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.7_seed0_matchClsfr_gapFeaOnly_bce_snglPrbPl_ep5.ckpt --eval



#PROB_NUM=150
#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--prob_word_type uda --prob_file_name $PROB_FILE_NAME --confid_prob $CONFID_PROB --dropout_prob 0.0 --prob_word_num $PROB_NUM \
#	--lambda-u 1 --T $T --wProb 1.0 \
#	--lrmain 0.000005 --lrlast $LRLAST --epochs 20 --val-iteration 1000 --specific_name _matchClsfr_gapFeaOnly_bce_snglPrbPl_prbFromUdaNoClsSep #--resume ./experiments/uda_probing_words/ag_news_10/ckpt/ --eval

#PROB_NUM=100
#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--prob_word_type uda --prob_file_name $PROB_FILE_NAME --confid_prob $CONFID_PROB --dropout_prob 0.0 --prob_word_num $PROB_NUM \
#	--lambda-u 1 --T $T --wProb 1.0 \
#	--lrmain 0.000005 --lrlast $LRLAST --epochs 20 --val-iteration 1000 --specific_name _matchClsfr_gapFeaOnly_bce_snglPrbPl_prbFromUdaNoClsSep #--resume ./experiments/uda_probing_words/ag_news_10/ckpt/ --eval

sleep 3h
PROB_NUM=50
python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
	--prob_word_type uda --prob_file_name $PROB_FILE_NAME --confid_prob $CONFID_PROB --dropout_prob 0.0 --prob_word_num $PROB_NUM \
	--lambda-u 1 --T $T --wProb 1.0 \
	--lrmain 0.000005 --lrlast $LRLAST --epochs 20 --val-iteration 1000 --specific_name _matchClsfr_gapFeaOnly_bce_snglPrbPl_prbFromUdaNoClsSep #--resume ./experiments/uda_probing_words/ag_news_10/ckpt/ --eval

PROB_NUM=200
#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--prob_word_type uda --prob_file_name $PROB_FILE_NAME --confid_prob $CONFID_PROB --dropout_prob 0.0 --prob_word_num $PROB_NUM \
#	--lambda-u 1 --T $T --wProb 1.0 \
#	--lrmain 0.000005 --lrlast $LRLAST --epochs 20 --val-iteration 1000 --specific_name _matchClsfr_gapFeaOnly_bce_snglPrbPl_prbFromUdaNoClsSep #--resume ./experiments/uda_probing_words/ag_news_10/ckpt/ --eval

# ---- lrlast=5e-04
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromuda_NoParallel_prbNum150_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.9_seed0_matchClsfr_gapFeaOnly_bce_snglPrbPl_ep4.ckpt
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromuda_NoParallel_prbNum150_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.8_seed0_matchClsfr_gapFeaOnly_bce_snglPrbPl_ep5.ckpt
# ---- lrlast=5e-06
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_5e-06_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromuda_NoParallel_prbNum150_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.8_seed0_matchClsfr_gapFeaOnly_bce_snglPrbPl_ep5.ckpt
#checkpoint.nU5000_wClsSep_wGap_lr5e-06_5e-06_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromuda_NoParallel_prbNum150_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.9_seed0_matchClsfr_gapFeaOnly_bce_snglPrbPl_ep8.ckpt
#ckpt/udaPrbWrd/checkpoint.nU5000_wClsSep_wGap_lr5e-06_5e-06_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_prbFromuda_NoParallel_prbNum150_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.7_seed0_matchClsfr_gapFeaOnly_bce_snglPrbPl_ep10.ckpt






