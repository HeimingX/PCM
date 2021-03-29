#!/usr/bin/env bash

# ============== original mixtext =====================
#python ./code/train.py --gpu 0,1,2 --n-labeled 10 \
#	--data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 4 --batch-size-u 8 --epochs 20 --val-iteration 1000 \
#	--lambda-u 1 --T 0.5 --alpha 16 --mix-layers-set 7 9 12 \
#	--lrmain 0.000005 --lrlast 0.0005

DATASET=./data/yahoo_answers_csv/yahoo_answers_csv/
DATASET=./data/ag_news_csv/

NLAB=5 #10
#NLAB=4 #10
LBS=4

#NLAB=3 #10
#LBS=3

#NLAB=2 #10
#NLAB=1 #10
#LBS=1

#GPU=0,2,3
GPU=0,1
NLAB=2 #10
RESUME=
python ./code/train.py --gpu $GPU --n-labeled $NLAB \
	--data-path $DATASET --batch-size $LBS --batch-size-u 8 --epochs 20 --val-iteration 1000 \
	--lambda-u 1 --T 0.5 --alpha 16 --mix-layers-set 7 9 12 \
	--lrmain 0.000005 --lrlast 0.0005 --resume $RESUME --eval

#./experiments/mixtext/yahoo_answers_2/ckpt/checkpoint.nU5000_lr5e-06_0.0005_ep20_it1000_lbs2_ubs8_mixMthd0_alpha16.0_wu1.0_T0.5_wuHng0_noClsSep_seed0_ep7.ckpt
NLAB=3 #10
#python ./code/train.py --gpu $GPU --n-labeled $NLAB \
#	--data-path $DATASET --batch-size $LBS --batch-size-u 8 --epochs 20 --val-iteration 1000 \
#	--lambda-u 1 --T 0.5 --alpha 16 --mix-layers-set 7 9 12 \
#	--lrmain 0.000005 --lrlast 0.0005 --resume ./experiments/mixtext/yahoo_answers_3/ckpt/checkpoint.nU5000_lr5e-06_0.0005_ep20_it1000_lbs3_ubs8_mixMthd0_alpha16.0_wu1.0_T0.5_wuHng0_noClsSep_seed0_ep9.ckpt --eval

NLAB=5 #10
#python ./code/train.py --gpu $GPU --n-labeled $NLAB \
#	--data-path $DATASET --batch-size $LBS --batch-size-u 8 --epochs 20 --val-iteration 1000 \
#	--lambda-u 1 --T 0.5 --alpha 16 --mix-layers-set 7 9 12 \
#	--lrmain 0.000005 --lrlast 0.0005 --specific_name _rerun2 --resume ./experiments/mixtext/yahoo_answers_5/ckpt/checkpoint.nU5000_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_mixMthd0_alpha16.0_wu1.0_T0.5_wuHng0_noClsSep_seed0_rerun2_ep16.ckpt --eval

#python ./code/train.py --gpu $GPU --n-labeled $NLAB \
#	--data-path $DATASET --batch-size $LBS --batch-size-u 8 --epochs 20 --val-iteration 1000 \
#	--lambda-u 1 --T 0.5 --alpha 16 --mix-layers-set 7 9 12 \
#	--lrmain 0.000005 --lrlast 0.0005 --specific_name _rerun --resume ./experiments/mixtext/yahoo_answers_5/ckpt/checkpoint.nU5000_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_mixMthd0_alpha16.0_wu1.0_T0.5_wuHng0_noClsSep_seed0_rerun_ep10.ckpt --eval

#experiments/mixtext/yahoo_answers_4/ckpt/checkpoint.nU5000_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_mixMthd0_alpha16.0_wu1.0_T0.5_wuHng0_noClsSep_seed0_ep11.ckpt
#experiments/mixtext/yahoo_answers_5/ckpt/checkpoint.nU5000_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_mixMthd0_alpha16.0_wu1.0_T0.5_wuHng0_noClsSep_seed0_ep6.ckpt

#python ./code/train.py --gpu 0 --n-labeled 10 \
#	--data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 4 --batch-size-u 8 --epochs 20 --val-iteration 1000 \
#	--lambda-u 1 --T 0.5 --alpha 16 --mix-layers-set 7 9 12 \
#	--lrmain 0.000005 --lrlast 0.0005 --eval --resume ./ckpt/mixtext/checkpoint.mixtext_ep10.ckpt

# =============== conditional s4l ====================
# ---------- change sharpen weight ---------
#python ./code/conditional_s4l_train.py --gpu 0,1,2 --n-labeled 10 \
#	--data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 3 --batch-size-u 6 --epochs 20 --val-iteration 1000 \
#	--lambda-u 1 --T 0.5 --alpha 16 --mix-layers-set 7 9 12 \
#	--lrmain 0.000005 --lrlast 0.0005 --lrmlm 1e-5 --mlm_batch_size 15

#python ./code/conditional_s4l_train.py --gpu 0,1,2 --n-labeled 10 \
#	--data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 3 --batch-size-u 6 --epochs 20 --val-iteration 1000 \
#	--lambda-u 1 --T 0.5 --alpha 16 --mix-layers-set 7 9 12 \
#	--lrmain 0.000005 --lrlast 0.0005 --lrmlm 1e-5 --mlm_batch_size 15 --eval --resume ./ckpt/conditionalS4l/checkpoint.nU5000_lr5e-06_0.0005_mlmLr1e-05_ep20_it1000_lbs3_ubs6_mlmbs15_ep11.ckpt

# ------------ 1st try ---------------
#python ./code/conditional_s4l_train.py --gpu 0,1,2 --n-labeled 10 \
#	--data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 3 --batch-size-u 6 --epochs 20 --val-iteration 1000 \
#	--lambda-u 1 --T 0.5 --alpha 16 --mix-layers-set 7 9 12 \
#	--lrmain 0.000005 --lrlast 0.0005 --mlm_batch_size 15

#python ./code/conditional_s4l_train.py --gpu 0,1,2 --n-labeled 10 \
#	--data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 3 --batch-size-u 6 --epochs 20 --val-iteration 1000 \
#	--lambda-u 1 --T 0.5 --alpha 16 --mix-layers-set 7 9 12 \
#	--lrmain 0.000005 --lrlast 0.0005 --mlm_batch_size 15 --resume ./ckpt/conditionalS4l/checkpoint.conditionalS4l

#python ./code/conditional_s4l_train.py --gpu 0,1,2 --n-labeled 10 \
#	--data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 3 --batch-size-u 6 --epochs 20 --val-iteration 1000 \
#	--lambda-u 1 --T 0.5 --alpha 16 --mix-layers-set 7 9 12 \
#	--lrmain 0.000005 --lrlast 0.0005 --mlm_batch_size 15 --eval --resume ./ckpt/conditionalS4l/checkpoint.conditionalS4l_ep12.ckpt
# ---------- 2ed try ------------
#python ./code/conditional_s4l_train.py --gpu 0,1,2 --n-labeled 10 \
#	--data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 3 --batch-size-u 6 --epochs 20 --val-iteration 1000 \
#	--lambda-u 1 --T 0.5 --alpha 16 --mix-layers-set 7 9 12 \
#	--lrmain 0.000005 --lrlast 0.0005 --mlm_batch_size 15

#python ./code/conditional_s4l_train.py --gpu 0,1,2 --n-labeled 10 \
#	--data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 3 --batch-size-u 6 --epochs 20 --val-iteration 1000 \
#	--lambda-u 1 --T 0.5 --alpha 16 --mix-layers-set 7 9 12 \
#	--lrmain 0.000005 --lrlast 0.0005 --mlm_batch_size 15 --resume ./ckpt/conditionalS4l/checkpoint.nU5000_lr5e-06_0.0005_mlmLr2e-05_ep20_it1000_lbs3_ubs6_mlmbs15_mixMthd0_alpha16.0_wu1.0_T0.5_wuHng0
# ---------- 3rd try ------------
#python ./code/conditional_s4l_train.py --gpu 0,1,2 --n-labeled 10 \
#	--data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 3 --batch-size-u 6 --epochs 20 --val-iteration 1000 \
#	--lambda-u 1 --T 0.5 --alpha 16 --mix-layers-set 7 9 12 \
#	--lrmain 0.000005 --lrlast 0.0005 --lrmlm 3e-5 --mlm_batch_size 18

#python ./code/conditional_s4l_train.py --gpu 0,1,2 --n-labeled 10 \
#	--data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 3 --batch-size-u 6 --epochs 20 --val-iteration 1000 \
#	--lambda-u 1 --T 0.5 --alpha 16 --mix-layers-set 7 9 12 \
#	--lrmain 0.000005 --lrlast 0.0005 --lrmlm 3e-5 --mlm_batch_size 18 --eval --resume ./ckpt/conditionalS4l/checkpoint.nU5000_lr5e-06_0.0005_mlmLr3e-05_ep20_it1000_lbs3_ubs6_mlmbs18_mixMthd0_alpha16.0_wu1.0_T0.5_wuHng0_ep19.ckpt
# ---------- 4th try ------------
#python ./code/conditional_s4l_train.py --gpu 0,1,2 --n-labeled 10 \
#	--data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 4 --batch-size-u 6 --epochs 20 --val-iteration 1000 \
#	--lambda-u 1 --T 0.5 --alpha 16 --mix-layers-set 7 9 12 \
#	--lrmain 0.000005 --lrlast 0.0005 --lrmlm 5e-5 --mlm_batch_size 27 --resume ./ckpt/conditionalS4l/checkpoint.nU5000_lr5e-06_0.0005_mlmLr5e-05_ep20_it1000_lbs4_ubs6_mlmbs27_mixMthd0_alpha16.0_wu1.0_T0.5_wuHng0

# ---------- no attention ---------
#python ./code/conditional_s4l_train.py --gpu 0,1,2 --n-labeled 10 \
#	--data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 3 --batch-size-u 6 --epochs 20 --val-iteration 1000 \
#	--lambda-u 1 --T 0.5 --alpha 16 --mix-layers-set 7 9 12 \
#	--lrmain 0.000005 --lrlast 0.0005 --mlm_batch_size 15 --specific_name _noAttention

#python ./code/conditional_s4l_train.py --gpu 0,1,2 --n-labeled 10 \
#	--data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 3 --batch-size-u 6 --epochs 20 --val-iteration 1000 \
#	--lambda-u 1 --T 0.5 --alpha 16 --mix-layers-set 7 9 12 \
#	--lrmain 0.000005 --lrlast 0.0005 --mlm_batch_size 15 --specific_name _noAttention --resume ./ckpt/conditionalS4lNoAttention/checkpoint.conditionalS4lNoAttention

#python ./code/conditional_s4l_train.py --gpu 0,1,2 --n-labeled 10 \
#	--data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 3 --batch-size-u 6 --epochs 20 --val-iteration 1000 \
#	--lambda-u 1 --T 0.5 --alpha 16 --mix-layers-set 7 9 12 \
#	--lrmain 0.000005 --lrlast 0.0005 --mlm_batch_size 15 --specific_name _noAttention --eval --resume ./ckpt/conditionalS4lNoAttention/checkpoint.conditionalS4lNoAttention_ep18.ckpt


# # debug
#python ./code/conditional_s4l_train.py --gpu 0,1,2 --n-labeled 10 \
#	--data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 4 --batch-size-u 8 --epochs 20 --val-iteration 1000 \
#	--lambda-u 1 --T 0.5 --alpha 16 --mix-layers-set 7 9 12 \
#	--lrmain 0.000005 --lrlast 0.0005

#python ./code/conditional_s4l_train.py --gpu 0 --n-labeled 10 \
#	--data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 1 --batch-size-u 1 --epochs 20 --val-iteration 1000 \
#	--lambda-u 1 --T 0.5 --alpha 16 --mix-layers-set 7 9 12 \
#	--lrmain 0.000005 --lrlast 0.0005 --mlm_batch_size=2


