import logging
import os
import random
import numpy as np
import torch
import ipdb


# ************************* define log *************************
class LogHandler:
    def __init__(self, run_context):
        self.run_context = run_context
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        self._log_to_file(formatter)
        self._log_to_terminal(formatter)

    def _log_to_file(self, formatter):
        log_file_name = self.run_context.exp_name + '.txt'
        fh = logging.FileHandler(os.path.join(self.run_context.result_dir, log_file_name))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def _log_to_terminal(self, formatter):
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)


def init_logger(run_context):
    global logger
    logger = LogHandler(run_context).logger


# ************************* run context *************************
class RunContext:
    """Creates directories and files for the run"""

    def __init__(self, runner_file, args):
        self.args = args
        if './code/normal_train.py' in runner_file:
            self.method_name = 'fine_tune'
        elif runner_file in ['./code/train.py', './code/train_pl1st.py', './code/train_imdb.py']:
            self.method_name = 'mixtext'
        elif './code/s4l_train.py' in runner_file:
            self.method_name = 's4l'
        elif './code/conditional_s4l_train.py' in runner_file:
            self.method_name = 'conditional_s4l'
        elif runner_file in ['./code/uda_with_probing_words.py',
                             './code/uda_with_probing_words_1vsall.py',
                             './code/uda_with_probing_words_1vsall_quant.py',
                             './code/uda_with_probing_words_1vsall_fixEpoch.py',
                             './code/uda_with_probing_words_1vsall_consist.py',
                             './code/uda_with_probing_words_1vsall_cos.py',
                             './code/uda_with_probing_words_split_match.py',
                             './code/uda_with_probing_words_1vsall_1clsfr.py',
                             './code/uda_with_probing_words_1vsall_2infer.py',
                             './code/uda_with_probing_words_1vsall_verifyUDA.py',
                             './code/uda_with_probing_words_1vsall_1clsfr_dist.py',
                             './code/uda_with_probing_words_1vsall_splitInfer.py',
                             './code/uda_with_probing_words_1vsall_2infer_pl1st.py',
                             './code/uda_with_probing_words_1vsall_pl1st.py',
                             './code/uda_with_probing_words_1vsall_pl1st_quant.py',
                             './code/uda_with_probing_words_1vsall_pl1st_consist.py',
                             './code/uda_with_probing_words_1vsall_pl1st_consist_noConfid.py',
                             './code/uda_with_probing_words_1vsall_pl1st_fixEpoch.py',
                             './code/uda_with_probing_words_1vsall_pl1st_agnews.py',
                             './code/uda_with_probing_words_1vsall_1clsfr_pl1st.py']:
            self.method_name = 'uda_probing_words'
        elif runner_file in ['./code/uda.py', './code/uda_bce.py', './code/uda_pl1st.py']:
            self.method_name = 'uda'
        elif runner_file in ['./code/ab_no_matching_clsfr_update_prob_words.py',
                             './code/ab_no_normal_clsfr_update_prob_words.py']:
            self.method_name = 'ablation_study'
        elif runner_file in ['./code/motivation_verify.py']:
            self.method_name = 'motivation_verify'
        else:
            raise LookupError

        if 'yahoo_answers' in args.data_path:
            dataset_name = 'yahoo_answers'
        elif 'ag_news' in args.data_path:
            dataset_name = 'ag_news'
        elif 'imdb' in args.data_path:
            dataset_name = 'imdb'
        elif 'dbpedia' in args.data_path:
            dataset_name = 'dbpedia'
        elif 'yelp' in args.data_path:
            if 'polarity' in args.data_path:
                dataset_name = 'yelp2'
            else:
                dataset_name = 'yelp5'
        elif 'amazon' in args.data_path:
            dataset_name = 'amazon'
        else:
            raise LookupError
        self.result_dir = "{root}/{method_name}/{dataset}_{num_label}".format(
            root='experiments',
            method_name=self.method_name,
            dataset=dataset_name,
            num_label=args.n_labeled
        )
        self.log_dir = os.path.join(self.result_dir, 'log')
        self.ckpt_dir = os.path.join(self.result_dir, 'ckpt')
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.exp_name = ''
        self.cur_exp_name()
        self.tensorboard_dir = os.path.join(self.result_dir, self.exp_name + '_tb')

    def cur_exp_name(self):

        if 'fine_tune' in self.method_name:
            self.exp_name += 'lr' + str(self.args.lrmain) + '_' + str(self.args.lrlast)
            self.exp_name += '_ep' + str(self.args.epochs)
            self.exp_name += '_bs' + str(self.args.batch_size)

            self.exp_name += '_wClsSep' if self.args.add_cls_sep else '_noClsSep'
            self.exp_name += '_noGap' if self.args.classify_with_cls else '_wGap'

        elif 'mixtext' in self.method_name:
            assert self.args.mix_option, 'mixtext method need set mix-option be True!!!'
            self.exp_name += 'nU' + str(self.args.un_labeled)
            self.exp_name += '_lr' + str(self.args.lrmain) + '_' + str(self.args.lrlast)
            self.exp_name += '_ep' + str(self.args.epochs) + '_it' + str(self.args.val_iteration)
            self.exp_name += '_lbs' + str(self.args.batch_size) + '_ubs' + str(self.args.batch_size_u)
            self.exp_name += '_mixMthd' + str(self.args.mix_method)
            self.exp_name += '_alpha' + str(self.args.alpha)
            self.exp_name += '_wu' + str(self.args.lambda_u)
            self.exp_name += '_T' + str(self.args.T)
            if self.args.lambda_u_hinge > 0:
                self.exp_name += '_entMrgn' + str(self.args.margin)
                self.exp_name += '_wuHng' + str(self.args.lambda_u_hinge)
            else:
                self.exp_name += '_wuHng0'
            self.exp_name += '_wClsSep' if self.args.add_cls_sep else '_noClsSep'

            # others are to be defined

        elif 'conditional_s4l' in self.method_name:
            assert self.args.mix_option, 'conditional_s4l method need set mix-option be True!!!'
            self.exp_name += 'nU' + str(self.args.un_labeled)
            self.exp_name += '_lr' + str(self.args.lrmain) + '_' + str(self.args.lrlast) \
                             + '_mlmLr' + str(self.args.lrmlm)
            self.exp_name += '_ep' + str(self.args.epochs) + '_it' + str(self.args.val_iteration)
            self.exp_name += '_lbs' + str(self.args.batch_size) + '_ubs' + str(self.args.batch_size_u) + \
                             '_mlmbs' + str(self.args.mlm_batch_size)
            self.exp_name += '_mixMthd' + str(self.args.mix_method)
            self.exp_name += '_alpha' + str(self.args.alpha)
            self.exp_name += '_wu' + str(self.args.lambda_u)
            self.exp_name += '_T' + str(self.args.T)
            if self.args.lambda_u_hinge > 0:
                self.exp_name += '_entMrgn' + str(self.args.margin)
                self.exp_name += '_wuHng' + str(self.args.lambda_u_hinge)
            else:
                self.exp_name += '_wuHng0'

        elif 'uda' in self.method_name or 'ablation_study' in self.method_name:
            self.exp_name += 'nU' + str(self.args.un_labeled)
            self.exp_name += '_wClsSep' if self.args.add_cls_sep else '_noClsSep'
            self.exp_name += '_noGap' if self.args.classify_with_cls else '_wGap'
            self.exp_name += '_lr' + str(self.args.lrmain) + '_' + str(self.args.lrlast)
            self.exp_name += '_ep' + str(self.args.epochs) + '_it' + str(self.args.val_iteration)
            self.exp_name += '_lbs' + str(self.args.batch_size) + '_ubs' + str(self.args.batch_size_u)
            self.exp_name += '_tsa' + self.args.tsa_type if self.args.tsa else '_noTsa'
            self.exp_name += '_wu' + str(self.args.lambda_u)
            self.exp_name += '_T' + str(self.args.T)
            self.exp_name += '_confid' + str(self.args.confid)

            if 'probing_words' in self.method_name or 'ablation_study' in self.method_name:
                if self.args.prob_word_type == 'dynamic':
                    if self.args.prob_file_name == 'cls_names':
                        self.exp_name += '_prbFromClsNames_' + self.args.prob_word_type
                    else:
                        if 'ft' in self.args.prob_file_name:
                            self.exp_name += '_prbFromFt_' + self.args.prob_word_type
                        elif 'uda' in self.args.prob_file_name:
                            self.exp_name += '_prbFromUDA_' + self.args.prob_word_type
                        elif 'mixtext' in self.args.prob_file_name:
                            self.exp_name += '_prbFromMixText_' + self.args.prob_word_type
                        else:
                            raise LookupError
                else:
                    self.exp_name += '_prbFrom' + self.args.prob_word_type
                self.exp_name += '_parallel' if self.args.parallel_prob else '_NoParallel'
                self.exp_name += '_prbNum' + str(self.args.prob_word_num)
                self.exp_name += '_wPrbWrd' + str(self.args.wProb)
                self.exp_name += '_prbLoss' + str(self.args.prob_loss_func)
                self.exp_name += '_cnfdPrb' + str(self.args.confid_prob)
                self.exp_name += '_multiSep' if self.args.multiple_sep else ''

        self.exp_name += '_seed' + str(self.args.seed)
        self.exp_name += '_seedl' + str(self.args.seed_l) if self.args.seed_l != 0 else ''
        if self.args.specific_name is not None:
            self.exp_name += self.args.specific_name


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False  # accelerate computing
    torch.backends.cudnn.deterministic = True  # avoid inference performance variation
