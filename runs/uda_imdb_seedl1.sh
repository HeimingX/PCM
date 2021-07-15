#!/usr/bin/env bash

LBS=4
UBS=8

T=1
DATASET=./data/imdb_csv/

LR_MAIN=5e-6
LR_LAST=5e-4
CONFID=0.95

SEEDL=0
# ============================== uda method ==============================
GPU=0,1,2
MAX_SEQ_LEN=512
NUL=12480

SAVE_SP=_pl1st_maxSeq${MAX_SEQ_LEN}_numUL${NUL}
SP_NAME=_pl1st_maxSeq${MAX_SEQ_LEN}_numUL${NUL}

# NLAB=50
# #PROB_TOPK=5
# python ./code/uda_pl1st.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
# 	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --seed_l $SEEDL \
# 	--lambda-u 1 --T $T --confid $CONFID \
# 	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--prob_topk $PROB_TOPK --resume ./experiments/uda/imdb_10/ckpt/best.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T1.0_confid0.95_seed0_pl1st.ckpt --eval

NLAB=20
#PROB_TOPK=30
python ./code/uda_pl1st.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
	--max_seq_len ${MAX_SEQ_LEN} --un-labeled ${NUL} \
	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --seed_l $SEEDL \
	--lambda-u 1 --T $T --confid $CONFID \
	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--prob_topk $PROB_TOPK --resume ./experiments/uda/imdb_5/ckpt/best.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T1.0_confid0.95_seed0_pl1st.ckpt --eval

# NLAB=10
# #PROB_TOPK=30
# python ./code/uda_pl1st.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
# 	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --seed_l $SEEDL \
# 	--lambda-u 1 --T $T --confid $CONFID \
# 	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--prob_topk $PROB_TOPK --resume ./experiments/uda/imdb_4/ckpt/best.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T1.0_confid0.95_seed0_pl1st.ckpt --eval

# NLAB=5
# #PROB_TOPK=30
# python ./code/uda_pl1st.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
# 	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --seed_l $SEEDL \
# 	--lambda-u 1 --T $T --confid $CONFID \
# 	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--prob_topk $PROB_TOPK --resume ./experiments/uda/imdb_3/ckpt/best.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T1.0_confid0.95_seed0_pl1st.ckpt --eval

# NLAB=3
# #PROB_TOPK=30
# python ./code/uda_pl1st.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
# 	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --seed_l $SEEDL \
# 	--lambda-u 1 --T $T --confid $CONFID \
# 	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--prob_topk $PROB_TOPK --resume ./experiments/uda/imdb_2/ckpt/best.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T1.0_confid0.95_seed0_pl1st.ckpt --eval

# ------- update probing words with consistent ratio on unlabeled validation data --------
PROB_FILE_NAME=cls_names
PROB_WORD_TYPE=dynamic
LOSS_FUNC=bce
#PROB_CONFID=0.7
#PROB_CONFID=0.8
PROB_CONFID=0.9
PROB_TOPK=30

PROB_NUM=30
SP_NAME=_consRatio #_maxSeq${MAX_SEQ_LEN}
SAVE_SP=_clsName_pn${PROB_NUM}_pc${PROB_CONFID}_consRatio #_maxSeq${MAX_SEQ_LEN}

#NLAB=50
#python ./code/uda_with_probing_words_1vsall_pl1st_consist.py --gpu $GPU --n-labeled $NLAB --add-cls-sep --max_seq_len ${MAX_SEQ_LEN} \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME \
#	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
#	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
#	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval

# NLAB=50
# python ./code/uda_with_probing_words_1vsall_pl1st_consist.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
# 	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --seed_l $SEEDL \
# 	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
# 	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
# 	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval

# NLAB=20
# python ./code/uda_with_probing_words_1vsall_pl1st_consist.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
# 	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --seed_l $SEEDL \
# 	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
# 	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
# 	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval

# NLAB=10
# python ./code/uda_with_probing_words_1vsall_pl1st_consist.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
# 	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --seed_l $SEEDL \
# 	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
# 	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
# 	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval

# NLAB=5
# python ./code/uda_with_probing_words_1vsall_pl1st_consist.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
# 	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --seed_l $SEEDL \
# 	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
# 	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
# 	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval

# NLAB=3
# python ./code/uda_with_probing_words_1vsall_pl1st_consist.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
# 	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --seed_l $SEEDL \
# 	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
# 	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
# 	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval

# ============================================================================================================
GPU=0
SEEDL=0
#MAX_SEQ_LEN=512

SAVE_SP=_pl1st #_maxSeq${MAX_SEQ_LEN}
SP_NAME=_pl1st #_maxSeq${MAX_SEQ_LEN}

# NLAB=50
# #PROB_TOPK=5
# python ./code/uda_pl1st.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
# 	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --seed_l $SEEDL \
# 	--lambda-u 1 --T $T --confid $CONFID \
# 	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--prob_topk $PROB_TOPK --resume ./experiments/uda/imdb_10/ckpt/best.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T1.0_confid0.95_seed0_pl1st.ckpt --eval

# NLAB=20
# #PROB_TOPK=30
# python ./code/uda_pl1st.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
# 	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --seed_l $SEEDL \
# 	--lambda-u 1 --T $T --confid $CONFID \
# 	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--prob_topk $PROB_TOPK --resume ./experiments/uda/imdb_5/ckpt/best.nU5000_wClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T1.0_confid0.95_seed0_pl1st.ckpt --eval


# PROB_FILE_NAME=cls_names
# PROB_WORD_TYPE=dynamic
# LOSS_FUNC=bce
# #PROB_CONFID=0.7
# #PROB_CONFID=0.8
# PROB_CONFID=0.9
# PROB_TOPK=30

# PROB_NUM=30
# SP_NAME=_consRatio #_maxSeq${MAX_SEQ_LEN}
# SAVE_SP=_clsName_pn${PROB_NUM}_pc${PROB_CONFID}_consRatio #_maxSeq${MAX_SEQ_LEN}

# NLAB=50
# python ./code/uda_with_probing_words_1vsall_pl1st_consist.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
# 	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --seed_l $SEEDL \
# 	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
# 	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
# 	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval

# NLAB=20
# python ./code/uda_with_probing_words_1vsall_pl1st_consist.py --gpu $GPU --n-labeled $NLAB --add-cls-sep \
# 	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --prob_file_name $PROB_FILE_NAME --seed_l $SEEDL \
# 	--prob_word_type $PROB_WORD_TYPE --prob_word_num $PROB_NUM --prob_topk $PROB_TOPK --parallel_prob --confid_prob $PROB_CONFID \
# 	--lambda-u 1 --T $T --wProb 1.0 --confid $CONFID --prob_loss_func $LOSS_FUNC \
# 	--lrmain $LR_MAIN --lrlast $LR_LAST --epochs 20 --val-iteration 1000 --specific_name $SP_NAME --prob_save_sp $SAVE_SP #--resume ./experiments/uda_probing_words/ #--eval
