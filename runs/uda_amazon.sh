#!/usr/bin/env bash

LBS=4
UBS=8

T=0.5 #1
DATASET=./data/amazon/

# ================================================ original uda =============================================
GPU=0 #,2
SEED=0

#CONFID=0.99
CONFID=0.95

#GPU=0,1,2 #2
GPU=2 #0
SP_NAME=_gapFeaOnly #_2gpu #_local_1gpu #_rerun3_again #2 #2
#sleep 2h
LR_MAIN=5e-6
#LR_MAIN=5e-4

#LR_LAST=5e-6
LR_LAST=5e-4
#LR_LAST=5e-3
#LR_LAST=5e-2

#NLAB=10
NLAB=5
python ./code/uda.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
	--lambda-u 1 --T $T --confid $CONFID \
	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME #--eval --resume ./experiments/uda/imdb_10/ckpt/best.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T1.0_confid0.95_seed0_gapFeaOnly.ckpt

NLAB=5
#T=2.5
#T=1.5
#python ./code/uda.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T --confid $CONFID \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _gapFeaOnly #--eval --resume ./experiments/uda/
#T=3
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
#LBS=2
#python ./code/uda.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T --confid $CONFID \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name _gapFeaOnly #--eval --resume ./experiments/uda/


# ============================== uda probing words ========================
#GPU=0,1,2 #1
#GPU=0,1 #1
GPU=0,1
#GPU=1
BEST_ACC=0.9025
PROB_FILE_NAME=uda_nlab10_top5_noStopWords_probWrods_ep18.npy
PROB_TOPK=5
PROB_NUM=50 #25 #50
T=1 #.5 #0.5 #0.5 #1 #2
CONFID=0.95 #5 #9 #0.99 #8 #9 #5 #9 #5 #9
PROB_CONFID=0.6 #9 #9 #5 #9 #91 #9 #92 #7 #3 #4 #3 #75 #8 #7 #8 #1
WU=1 #0.5
WUP=1 #0.5
PROB_WORD_TYPE=dynamic

SP_NAME=_onlyMatchClsfr_2gpu #_test #_noMatchLoss_noProb #_rerun10 #9 #8 #7 #6 # 5 #4 #3 #2 #1 #_gtVerify
SAVE_SP=_new1 #2 #1

NLAB=10
#python ./code/uda_with_probing_words_1vsall_1clsfr.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type dynamic --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --init_best_acc $BEST_ACC --parallel_prob --confid_prob $PROB_CONFID --dropout_prob 0.0 \
#	--lambda-u 1 --T $T --wProb $WUP --confid $CONFID --lambda-u $WU \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/imdb_10/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T1.0_confid0.99_prbFromUDA_dynamic_parallel_prbNum50_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.98_seed0_onlyMatchClsfr.ckpt #--eval

#python ./code/uda_with_probing_words_1vsall_1clsfr_dist.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type dynamic --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --init_best_acc $BEST_ACC --parallel_prob --confid_prob $PROB_CONFID --dropout_prob 0.0 \
#	--lambda-u 1 --T $T --wProb $WUP --confid $CONFID --lambda-u $WU \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/imdb_10/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T1.0_confid0.99_prbFromUDA_dynamic_parallel_prbNum50_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.98_seed0_onlyMatchClsfr.ckpt #--eval

#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type dynamic --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --init_best_acc $BEST_ACC --parallel_prob --confid_prob $PROB_CONFID --dropout_prob 0.0 \
#	--lambda-u 1 --T $T --wProb $WUP --confid $CONFID --lambda-u $WU \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/imdb_10/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T1.0_confid0.99_prbFromUDA_dynamic_parallel_prbNum50_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.98_seed0_onlyMatchClsfr.ckpt #--eval

#SP_NAME=_onlyMatchClsfr_noMatchLoss #_rerun10 #9 #8 #7 #6 # 5 #4 #3 #2 #1 #_gtVerify
#SP_NAME=_originClsfr_noProbClsfr #_rerun10 #9 #8 #7 #6 # 5 #4 #3 #2 #1 #_gtVerify
#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --init_best_acc $BEST_ACC --parallel_prob --confid_prob $PROB_CONFID --dropout_prob 0.0 \
#	--lambda-u 1 --T $T --wProb $WUP --confid $CONFID --lambda-u $WU \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/imdb_10/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T1.0_confid0.99_prbFromUDA_dynamic_parallel_prbNum50_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.98_seed0_onlyMatchClsfr.ckpt #--eval

#GPU=0,2
GPU=0,1
SP_NAME=_2infer #_verifyUDA #_rerun10 #9 #8 #7 #6 # 5 #4 #3 #2 #1 #_gtVerify
#LOSS_FUNC=ce
#CONFID=0.95
PROB_CONFID=0.95
#LR_LAST=5e-3
#LR_LAST=5e-4
LR_MAIN=5e-6

PROB_NUM=200
LOSS_FUNC=bce
CONFID=0.95
PROB_CONFID=0.8
#LR_LAST=5e-3
LR_LAST=5e-4
sleep 6h
python ./code/uda_with_probing_words_1vsall_2infer.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --init_best_acc $BEST_ACC --parallel_prob --confid_prob $PROB_CONFID --dropout_prob 0.0 --feed-prob \
	--lambda-u 1 --T $T --wProb $WUP --confid $CONFID --lambda-u $WU --prob_loss_func $LOSS_FUNC \
	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/imdb_10/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T1.0_confid0.99_prbFromUDA_dynamic_parallel_prbNum50_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.98_seed0_onlyMatchClsfr.ckpt #--eval

SP_NAME=_originClsfr_ProbClsfr #_rerun10 #9 #8 #7 #6 # 5 #4 #3 #2 #1 #_gtVerify
#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --init_best_acc $BEST_ACC --parallel_prob --confid_prob $PROB_CONFID --dropout_prob 0.0 --feed-prob \
#	--lambda-u 1 --T $T --wProb $WUP --confid $CONFID --lambda-u $WU \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/imdb_10/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T1.0_confid0.99_prbFromUDA_dynamic_parallel_prbNum50_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.98_seed0_onlyMatchClsfr.ckpt #--eval

#python ./code/uda_with_probing_words_1vsall.py --gpu $GPU --n-labeled $NLAB --add-cls-sep --multiple-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type dynamic --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --init_best_acc $BEST_ACC --parallel_prob --confid_prob $PROB_CONFID --dropout_prob 0.0 \
#	--lambda-u 1 --T $T --wProb $WUP --confid $CONFID --lambda-u $WU \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/imdb_10/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T1.0_confid0.99_prbFromUDA_dynamic_parallel_prbNum50_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.98_seed0_onlyMatchClsfr.ckpt #--eval

GPU=0,1 #1
GPU=2,0 #1
SP_NAME=_verifyUDA #_rerun10 #9 #8 #7 #6 # 5 #4 #3 #2 #1 #_gtVerify
PROB_WORD_TYPE=None
CONFID=0.95 #5 #9 #0.99 #8 #9 #5 #9 #5 #9
LR_LAST=5e-4
#LR_LAST=5e-5
#python ./code/uda_with_probing_words_1vsall_verifyUDA.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --init_best_acc $BEST_ACC --confid_prob $PROB_CONFID --dropout_prob 0.0 \
#	--lambda-u 1 --T $T --wProb $WUP --confid $CONFID --lambda-u $WU \
#	--lrmain 0.000005 --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/imdb_10/ckpt/checkpoint.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T1.0_confid0.99_prbFromUDA_dynamic_parallel_prbNum50_drpPrb0.0_wPrbWrd1.0_cnfdPrb0.98_seed0_onlyMatchClsfr.ckpt #--eval

# ============================== dynamic probing words ========================
GPU=0,1
BEST_ACC=0
PROB_FILE_NAME=cls_names
PROB_TOPK=30
PROB_NUM=75
T=2
CONFID=0.99
PROB_CONFID=0.9

SP_NAME=_onlyMatchClsfr
SAVE_SP=_new

NLAB=10
#python ./code/uda_with_probing_words_1vsall_1clsfr.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type dynamic --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --init_best_acc $BEST_ACC --parallel_prob --confid_prob $PROB_CONFID --dropout_prob 0.0 \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval

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






/home/haiming/fast_hm/code/rtx148/MixText/

/home/haiming/fast/MixText/data/
