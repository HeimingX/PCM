#!/usr/bin/env bash

SEED=514
SEED=414
SEED=314
SEED=214
SEED=111

# basic case
#python ./code/text_classification_yh.py --gpu 1 --seed $SEED \
#	--ds_name 'AG_NEWs' --n-labeled 5 --batch-size 4 --epochs 10 \
#	--opt-name 'Adam' --lrmain 2e-5 --lrlast -1 \
#	--max-seq-len 0 --add-cls-sep --classify-with-cls --use-cls-type 'qy'

# ab1: diff cls type
#python ./code/text_classification_yh.py --gpu 2 --seed $SEED \
#	--ds_name 'AG_NEWs' --n-labeled 5 --batch-size 4 --epochs 10 \
#	--opt-name 'Adam' --lrmain 2e-5 --lrlast -1 \
#	--max-seq-len 0 --add-cls-sep --classify-with-cls --use-cls-type 'normal'

# ab2: use gap
#python ./code/text_classification_yh.py --gpu 0 --seed $SEED \
#	--ds_name 'AG_NEWs' --n-labeled 5 --batch-size 4 --epochs 10 \
#	--opt-name 'Adam' --lrmain 2e-5 --lrlast -1 \
#	--max-seq-len 0 --add-cls-sep

# ab2.1: use gap without [CLS]&[SEP] token
#python ./code/text_classification_yh.py --gpu 2 --seed $SEED \
#	--ds_name 'AG_NEWs' --n-labeled 5 --batch-size 4 --epochs 10 \
#	--opt-name 'Adam' --lrmain 2e-5 --lrlast -1 \
#	--max-seq-len 0

# ab2.2: use gap filter [CLS]&[SEP]'s feature
#python ./code/text_classification_yh.py --gpu 1 --seed $SEED \
#	--ds_name 'AG_NEWs' --n-labeled 5 --batch-size 4 --epochs 10 \
#	--opt-name 'Adam' --lrmain 2e-5 --lrlast 2e-5 \
#	--max-seq-len 0 --add-cls-sep --specific_name _noCLSSEPfea 

# ab2.2.1: use gap filter [CLS]&[SEP]'s feature, diff lr
#python ./code/text_classification_yh.py --gpu 1 --seed $SEED \
#	--ds_name 'AG_NEWs' --n-labeled 5 --batch-size 4 --epochs 10 \
#	--opt-name 'Adam' --lrmain 1e-5 --lrlast 1e-3 \
#	--max-seq-len 0 --add-cls-sep --specific_name _noCLSSEPfea 

# ab2.2.1.1: use gap filter [CLS]&[SEP]'s feature, diff lr, training longer
#python ./code/text_classification_yh.py --gpu 1 --seed $SEED \
#	--ds_name 'AG_NEWs' --n-labeled 5 --batch-size 4 --epochs 20 \
#	--opt-name 'Adam' --lrmain 1e-5 --lrlast 1e-3 \
#	--max-seq-len 0 --add-cls-sep --specific_name _noCLSSEPfea 

# ab3: fix seq len
#python ./code/text_classification_yh.py --gpu 1 --seed $SEED \
#	--ds_name 'AG_NEWs' --n-labeled 5 --batch-size 4 --epochs 10 \
#	--opt-name 'Adam' --lrmain 2e-5 --lrlast -1 \
#	--max-seq-len 256 --add-cls-sep

# ab4: diff optim: adam vs. adamW
#python ./code/text_classification_yh.py --gpu 0 --seed $SEED \
#	--ds_name 'AG_NEWs' --n-labeled 5 --batch-size 4 --epochs 10 \
#	--opt-name 'AdamW' --lrmain 2e-5 --lrlast 2e-5 \
#	--max-seq-len 0 --add-cls-sep

# ab4.1: diff lr: 1e-5, 1e-3
#python ./code/text_classification_yh.py --gpu 2 --seed $SEED \
#	--ds_name 'AG_NEWs' --n-labeled 5 --batch-size 4 --epochs 10 \
#	--opt-name 'AdamW' --lrmain 1e-5 --lrlast 1e-3 \
#	--max-seq-len 0 --add-cls-sep

# ab2.2: diff lr: 1e-5, 1e-3
#python ./code/text_classification_yh.py --gpu 2 --seed $SEED \
#	--ds_name 'AG_NEWs' --n-labeled 5 --batch-size 4 --epochs 10 \
#	--opt-name 'Adam' --lrmain 1e-5 --lrlast 1e-3 \
#	--max-seq-len 0 --add-cls-sep

# ab5.0: add probing word: NEWS
#python ./code/text_classification_yh.py --gpu 0 --seed $SEED \
#	--ds_name 'AG_NEWs' --n-labeled 5 --batch-size 4 --epochs 10 \
#	--opt-name 'Adam' --lrmain 1e-5 --lrlast 1e-3 \
#	--max-seq-len 0 --add-cls-sep --add_prob

# ab5.0.1: add probing word: NEWS
#python ./code/text_classification_yh.py --gpu 1 --seed $SEED \
#	--ds_name 'AG_NEWs' --n-labeled 5 --batch-size 4 --epochs 10 \
#	--opt-name 'Adam' --lrmain 1e-5 --lrlast 1e-3 \
#	--max-seq-len 0 --add-cls-sep --add_prob --specific_name _noPrb4Tst

# ab5.1: add probing word: CLASS
#python ./code/text_classification_yh.py --gpu 2 --seed $SEED \
#	--ds_name 'AG_NEWs' --n-labeled 5 --batch-size 4 --epochs 10 \
#	--opt-name 'Adam' --lrmain 1e-5 --lrlast 1e-3 \
#	--max-seq-len 0 --add-cls-sep --add_prob --prob_word_type 'CLASS'

# ab5.1.1: add probing word: CLASS
#python ./code/text_classification_yh.py --gpu 2 --seed $SEED \
#	--ds_name 'AG_NEWs' --n-labeled 5 --batch-size 4 --epochs 10 \
#	--opt-name 'Adam' --lrmain 1e-5 --lrlast 1e-3 \
#	--max-seq-len 0 --add-cls-sep --add_prob --prob_word_type 'CLASS' --specific_name _noPrb4Tst

# ab5.2: add probing word: ClassWise
#python ./code/text_classification_yh.py --gpu 1 --seed $SEED \
#	--ds_name 'AG_NEWs' --n-labeled 5 --batch-size 4 --epochs 10 \
#	--opt-name 'Adam' --lrmain 1e-5 --lrlast 1e-3 \
#	--max-seq-len 0 --add-cls-sep --add_prob --prob_word_type 'ClassWise'

# ab5.3: add probing word: LabList
#python ./code/text_classification_yh.py --gpu 1 --seed $SEED \
#	--ds_name 'AG_NEWs' --n-labeled 5 --batch-size 4 --epochs 10 \
#	--opt-name 'Adam' --lrmain 1e-5 --lrlast 1e-3 \
#	--max-seq-len 0 --add-cls-sep --add_prob --prob_word_type 'LabList'

# ab5.3.1: add probing word: LabList, average prob word embedding
#python ./code/text_classification_yh.py --gpu 2 --seed $SEED \
#	--ds_name 'AG_NEWs' --n-labeled 5 --batch-size 4 --epochs 10 \
#	--opt-name 'Adam' --lrmain 1e-5 --lrlast 1e-3 \
#	--max-seq-len 0 --add-cls-sep --add_prob --prob_word_type 'LabList' --specific_name _avgPrbEmbedReRun

# ab5.4: add probing word: SimList
#python ./code/text_classification_yh.py --gpu 0 --seed $SEED \
#	--ds_name 'AG_NEWs' --n-labeled 5 --batch-size 4 --epochs 10 \
#	--opt-name 'Adam' --lrmain 1e-5 --lrlast 1e-3 \
#	--max-seq-len 0 --add-cls-sep --add_prob --prob_word_type 'SimList'

# ab5.4.1: add probing word: SimList, average prob word embedding
#python ./code/text_classification_yh.py --gpu 0 --seed $SEED \
#	--ds_name 'AG_NEWs' --n-labeled 5 --batch-size 4 --epochs 10 \
#	--opt-name 'Adam' --lrmain 1e-5 --lrlast 1e-3 \
#	--max-seq-len 0 --add-cls-sep --add_prob --prob_word_type 'SimList' --specific_name _avgPrbEmbed

# ab5.4.1.1: add probing word: SimList, average prob word embedding, no prob
# word for test, tag false to noPrb4Tst, should be noCLSSEPfea
#python ./code/text_classification_yh.py --gpu 1 --seed $SEED \
#	--ds_name 'AG_NEWs' --n-labeled 5 --batch-size 4 --epochs 10 \
#	--opt-name 'Adam' --lrmain 1e-5 --lrlast 1e-3 \
#	--max-seq-len 0 --add-cls-sep --add_prob --prob_word_type 'SimList'	--specific_name _avgPrbEmbed_noCLSSEPfea #noPrb4Tst

# ab5.4.1.2: add probing word: SimList, average prob word embedding, no prob
# word for test,  should be noCLSSEPfea
#python ./code/text_classification_yh.py --gpu 0 --seed $SEED \
#	--ds_name 'AG_NEWs' --n-labeled 5 --batch-size 4 --epochs 10 \
#	--opt-name 'Adam' --lrmain 1e-5 --lrlast 1e-3 \
#	--max-seq-len 0 --add-cls-sep --add_prob --prob_word_type 'SimList' --specific_name _avgPrbEmbed_noCLSSEPPrbfea

# ab5.4.1.2: add probing word: SimList, average prob word embedding, using
# averaged probing word feature to classify
#python ./code/text_classification_yh.py --gpu 1 --seed $SEED \
#	--ds_name 'AG_NEWs' --n-labeled 5 --batch-size 4 --epochs 10 \
#	--opt-name 'Adam' --lrmain 1e-5 --lrlast 1e-3 \
#	--max-seq-len 0 --add-cls-sep --add_prob --prob_word_type 'SimList' --specific_name _avgPrbEmbed_wPrbClsfr

# ab5.4.1.2: add probing word: SimList, average prob word embedding, using
# averaged probing word feature to classify
python ./code/text_classification_yh.py --gpu 1 --seed $SEED \
	--ds_name 'AG_NEWs' --n-labeled 5 --batch-size 4 --epochs 10 \
	--opt-name 'Adam' --lrmain 1e-5 --lrlast 1e-3 \
	--max-seq-len 0 --add-cls-sep --add_prob --prob_word_type 'SimList' --avg_prob_at 'output' --classify-with 'avgPrb'

