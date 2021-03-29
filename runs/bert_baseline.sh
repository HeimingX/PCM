#!/usr/bin/env bash

#python ./code/normal_train.py --gpu 0,1 --n-labeled 10 --data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 8 --epochs 20 
#python ./code/normal_train.py --gpu 2 --n-labeled 10 --data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 8 --epochs 20
#python ./code/normal_train.py --gpu 0,1 --n-labeled 10 --data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 8 --epochs 20 --specific_name=_wAttentionMask

# fine-tuning with cls and sep token
#python ./code/normal_train.py --gpu 1,2 --n-labeled 10 --data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --lrmain=6e-5 --lrlast=6e-5 --batch-size 8 --epochs 20 --add-cls-sep --resume=./ckpt/baseline/checkpoint.lr6e-05_6e-05_ep20_bs8_wClsSep_wGap_ep4.ckpt --eval

#python ./code/normal_train.py --gpu 1,2 --n-labeled 10 --data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --lrmain=7e-5 --lrlast=7e-5 --batch-size 8 --epochs 20 --add-cls-sep --resume=./ckpt/baseline/checkpoint.lr7e-05_7e-05_ep20_bs8_wClsSep_wGap_ep19.ckpt --eval
#python ./code/normal_train.py --gpu 2 --n-labeled 10 --data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 8 --epochs 20 --add-cls-sep --classify-with-cls --specific_name=_run2
#python ./code/normal_train.py --gpu 1 --n-labeled 10 --data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --lrmain=1e-5 --lrlast=2e-3 --batch-size 8 --epochs 20 --add-cls-sep --classify-with-cls #--specific_name=_drpoutLinear
#python ./code/normal_train.py --gpu 2 --n-labeled 10 --data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --lrmain=1e-5 --lrlast=5e-3 --batch-size 8 --epochs 20 --add-cls-sep --classify-with-cls #--specific_name=_drpoutLinear
#python ./code/normal_train.py --gpu 0 --n-labeled 10 --data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --lrmain=1e-5 --lrlast=5e-4 --batch-size 8 --epochs 20 --add-cls-sep --classify-with-cls #--specific_name=_drpoutLinear
#python ./code/normal_train.py --gpu 0,1 --n-labeled 10 --data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --lrmain=2e-5 --lrlast=2e-5 --batch-size 8 --epochs 20 --add-cls-sep --classify-with-cls --specific_name=_drpoutLinear
#python ./code/normal_train.py --gpu 0 --n-labeled 10 --data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --lrmain=2e-5 --lrlast=2e-5 --batch-size 8 --epochs 50 --add-cls-sep --classify-with-cls --specific_name=_drpoutLinear
#python ./code/normal_train.py --gpu 1,2 --n-labeled 10 --data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --lrmain=6e-5 --lrlast=6e-5 --batch-size 8 --epochs 20 --add-cls-sep --classify-with-cls --resume=./ckpt/baseline/checkpoint.lr6e-05_6e-05_ep20_bs8_wClsSep_noGap_ep15.ckpt --eval #--specific_name=_drpoutLinear
#python ./code/normal_train.py --gpu 1,2 --n-labeled 10 --data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --lrmain=7e-5 --lrlast=7e-5 --batch-size 8 --epochs 20 --add-cls-sep --classify-with-cls --resume=./ckpt/baseline/checkpoint.lr7e-05_7e-05_ep20_bs8_wClsSep_noGap_ep5.ckpt --eval #--specific_name=_drpoutLinear

#python ./code/normal_train.py --gpu 1 --n-labeled 10 --data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --lrmain=5e-5 --lrlast=5e-3 --batch-size 8 --epochs 50 --add-cls-sep --classify-with-cls #--specific_name=_drpoutLinear
#python ./code/normal_train.py --gpu 1 --n-labeled 10 --data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --lrmain=2e-5 --lrlast=2e-4 --batch-size 8 --epochs 20 --add-cls-sep --classify-with-cls --specific_name=_drpoutLinear
#python ./code/normal_train.py --gpu 2,1 --n-labeled 10 --data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 8 --epochs 20 --add-cls-sep --specific_name=_drpoutLinear

# add probing words
#python ./code/probing_word_train.py --gpu 0 --n-labeled 10 --data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 8 --epochs 20 --add_prob
#python ./code/probing_word_train.py --gpu 0,1 --n-labeled 10 --data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 8 --epochs 20 --add_prob --resume ./ckpt/baseline_prob_word/checkpoint.baseline_prob_word

#python ./code/normal_train.py --gpu 0 --n-labeled 10 --data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 8 --epochs 20 --lrmain=6e-5 --lrlast=6e-5 --add_prob --add-cls-sep --classify-with-cls
#python ./code/normal_train.py --gpu 0,2 --n-labeled 10 --data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 8 --epochs 20 --lrmain=1e-5 --lrlast=1e-3 --add_prob --add-cls-sep --prob_word_type 'CLASS' --resume=./ckpt/baseline/checkpoint.lr1e-05_0.001_ep20_bs8_wClsSep_wGap_wPrbngWrd_ep19.ckpt --eval
#python ./code/normal_train.py --gpu 1,2 --n-labeled 10 --data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 8 --epochs 20 --lrmain=1e-5 --lrlast=1e-3 --add_prob --add-cls-sep --prob_word_type 'NEWS' --resume=./ckpt/baseline/checkpoint.lr1e-05_0.001_ep20_bs8_wClsSep_wGap_wPrbngWrdNEWS_ep16.ckpt --eval
#python ./code/normal_train.py --gpu 1,2 --n-labeled 10 --data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 8 --epochs 20 --lrmain=1e-5 --lrlast=1e-3 --add_prob --add-cls-sep --prob_word_type 'LabList' --resume=./ckpt/baseline/checkpoint.lr1e-05_0.001_ep20_bs8_wClsSep_wGap_wPrbngWrdLabList_ep19.ckpt --eval
python ./code/normal_train.py --gpu 1,2 --n-labeled 10 --data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 8 --epochs 20 --lrmain=1e-5 --lrlast=1e-3 --add_prob --add-cls-sep --prob_word_type 'SimList' --resume=./ckpt/baseline/checkpoint.lr1e-05_0.001_ep20_bs8_wClsSep_wGap_wPrbngWrdSimList_ep19.ckpt --eval

#python ./code/normal_train.py --gpu 2 --n-labeled 10 --data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 8 --epochs 20 --lrmain=7e-5 --lrlast=7e-5 --add_prob --add-cls-sep #--resume=./ckpt/baseline/checkpoint.lr6e-05_6e-05_ep20_bs8_wClsSep_wGap_ep4.ckpt --eval


# mlm-pretrain+fine-tuning
#python ./code/normal_train.py --gpu 2,1 --n-labeled 10 --data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 8 --epochs 20 --resume ./ckpt/pretrain/checkpoint.pretrain_ep19.ckpt
#python ./code/normal_train.py --gpu 1 --n-labeled 10 --data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 8 --epochs 20 --lrmain=7e-5 --lrlast=7e-5 --add-cls-sep --resume ./ckpt/pretrain/checkpoint.pretrain_ep19.ckpt --specific_name=_fromSlfPreTrain
#python ./code/normal_train.py --gpu 0 --n-labeled 10 --data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 8 --epochs 20 --lrmain=7e-5 --lrlast=7e-5 --add_prob --add-cls-sep --resume ./ckpt/pretrain/checkpoint.pretrain_ep19.ckpt --specific_name=_fromSlfPreTrain
#python ./code/normal_train.py --gpu 1 --n-labeled 10 --data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 8 --epochs 20 --lrmain=6e-5 --lrlast=6e-5 --add-cls-sep --classify-with-cls --resume ./ckpt/pretrain/checkpoint.pretrain_ep19.ckpt --specific_name=_fromSlfPreTrain
#python ./code/normal_train.py --gpu 2 --n-labeled 10 --data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 8 --epochs 20 --lrmain=6e-5 --lrlast=6e-5 --add_prob --add-cls-sep --classify-with-cls --resume ./ckpt/pretrain/checkpoint.pretrain_ep19.ckpt --specific_name=_fromSlfPreTrain
#python ./code/normal_train.py --gpu 2 --n-labeled 10 --data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 8 --epochs 20 --resume ./ckpt/pretrain/checkpoint.pretrain_ep19.ckpt --specific_name=_fromSlfPreTrain
#python ./code/normal_train.py --gpu 1 --n-labeled 10 --data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --batch-size 8 --epochs 20 --add_prob --resume ./ckpt/pretrain/checkpoint.pretrain_ep19.ckpt --specific_name=_fromSlfPreTrain


#python ./code/normal_train.py --gpu 1 --n-labeled 10 --data-path ./data/yahoo_answers_csv/yahoo_answers_csv/ --lrmain=7e-5 --lrlast=7e-5 --batch-size 8 --epochs 20 --add-cls-sep --resume=./ckpt/baseline/checkpoint.lr7e-05_7e-05_ep20_bs8_wClsSep_wGap_ep19.ckpt --eval






















