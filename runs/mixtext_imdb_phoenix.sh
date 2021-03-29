#!/bin/bash

# Configure the resources required
#SBATCH -p batch                                                # partition (this is the queue your job will be added to)
#SBATCH -n 1              	                                # number of tasks (sequential job starts 1 task) (check this if your job unexpectedly uses 2 nodes)
#SBATCH -c 4              	                                # number of cores (sequential job calls a multi-thread program that uses 8 cores)
#SBATCH --time=24:00:00                                         # time allocation, which has the format (D-HH:MM), here set to 1 hour
#SBATCH --gres=gpu:2                                            # generic resource required (here requires 4 GPUs)
#SBATCH --mem=80GB                                              # specify memory required per node (here set to 16 GB)
#SBATCH -M volta

# Configure notifications
#SBATCH --mail-type=END                                         # Send a notification email when the job is done (=END)
#SBATCH --mail-type=FAIL                                        # Send a notification email when the job fails (=FAIL)
#SBATCH --mail-user=hai-ming.xu@adelaide.edu.au          # Email to which notifications will be sent

# Execute your script (due to sequential nature, please select proper compiler as your script corresponds to)
#bash ./run.sh  # bash script used here for demonstration purpose, you should select proper compiler for your needs
source activate py3.6_pt1.4
nvidia-smi

# ============== original mixtext =====================
#python ./code/train.py --gpu 0,1,2 --n-labeled 10 \
#	--data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 4 --batch-size-u 8 --epochs 20 --val-iteration 1000 \
#	--lambda-u 1 --T 0.5 --alpha 16 --mix-layers-set 7 9 12 \
#	--lrmain 0.000005 --lrlast 0.0005

LBS=4 #8 #16 #32
UBS=8 #18 #24 #30 #24 #48 #96

GPU=0,1 #,2,3
DATASET=./data/imdb_csv/
#SP_NAME=_rerun2
SP_NAME=_run1
T=1
NLAB=10
python ./code/train.py --gpu $GPU --n-labeled $NLAB \
	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --epochs 20 --val-iteration 1000 \
	--lambda-u 1 --T $T --alpha 16 --mix-layers-set 7 9 12 \
	--lrmain 0.000005 --lrlast 0.0005 --specific_name $SP_NAME #--resume ./experiments/mixtext/ --eval

NLAB=5
#SP_NAME=_run2
python ./code/train.py --gpu $GPU --n-labeled $NLAB \
	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --epochs 20 --val-iteration 1000 \
	--lambda-u 1 --T $T --alpha 16 --mix-layers-set 7 9 12 \
	--lrmain 0.000005 --lrlast 0.0005 --specific_name $SP_NAME #--resume ./experiments/mixtext/ --eval

NLAB=4
python ./code/train.py --gpu $GPU --n-labeled $NLAB \
	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --epochs 20 --val-iteration 1000 \
	--lambda-u 1 --T $T --alpha 16 --mix-layers-set 7 9 12 \
	--lrmain 0.000005 --lrlast 0.0005 --specific_name $SP_NAME #--resume ./experiments/mixtext/ --eval

NLAB=3
#python ./code/train.py --gpu $GPU --n-labeled $NLAB \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --epochs 20 --val-iteration 1000 \
#	--lambda-u 1 --T $T --alpha 16 --mix-layers-set 7 9 12 \
#	--lrmain 0.000005 --lrlast 0.0005 --specific_name $SP_NAME #--resume ./experiments/mixtext/ --eval

NLAB=2
#python ./code/train.py --gpu $GPU --n-labeled $NLAB \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --epochs 20 --val-iteration 1000 \
#	--lambda-u 1 --T $T --alpha 16 --mix-layers-set 7 9 12 \
#	--lrmain 0.000005 --lrlast 0.0005 --specific_name $SP_NAME #--resume ./experiments/mixtext/ --eval

NLAB=1
#python ./code/train.py --gpu $GPU --n-labeled $NLAB \
#	--data-path $DATASET --batch-size $LBS --batch-size-u $UBS --epochs 20 --val-iteration 1000 \
#	--lambda-u 1 --T $T --alpha 16 --mix-layers-set 7 9 12 \
#	--lrmain 0.000005 --lrlast 0.0005 --specific_name $SP_NAME #--resume ./experiments/mixtext/ --eval

# ================ original uda ========================
#T=0.5
#python ./code/uda.py --gpu $GPU --n-labeled $NLAB \
#	--data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size $LBS --batch-size-u $UBS \
#	--lambda-u 1 --T $T \
#	--lrmain 0.000005 --lrlast 0.0005 --epochs 20 --val-iteration 1000 #--eval --resume ./ckpt/uda/


#checkpoint.nU5000_noClsSep_wGap_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_noTsa_wu1.0_T0.5_confid0.95_seed0_ep16.ckpt

#ag_news_10/ckpt/checkpoint.nU5000_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_mixMthd0_alpha16.0_wu1.0_T0.5_wuHng0_seed0_ep14.ckpt
#ckpt/mixtext/checkpoint.nU5000_lr5e-06_0.0005_ep20_it1000_lbs4_ubs8_mixMthd0_alpha16.0_wu1.0_T0.5_wuHng0_ep9.ckpt

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


