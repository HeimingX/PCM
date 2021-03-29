import argparse
import os
import random
import math
import time
import logging
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from pytorch_transformers import *
from torch.autograd import Variable
from torch.utils.data import Dataset
from tensorboardX import SummaryWriter

import utils, global_file, retrieval_attention
import ipdb

parser = argparse.ArgumentParser(description='PyTorch UDA with pre-defined probing words Models')

parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch-size', default=4, type=int, metavar='N', help='train batchsize')
parser.add_argument('--batch-size-u', default=24, type=int, metavar='N', help='train batchsize')

parser.add_argument('--lrmain', '--learning-rate-bert', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate for bert')
parser.add_argument('--lrlast', '--learning-rate-model', default=0.001, type=float,
                    metavar='LR', help='initial learning rate for models')

parser.add_argument('--gpu', default='0,1,2,3', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--n-labeled', type=int, default=20, help='Number of labeled data')
parser.add_argument('--un-labeled', default=5000, type=int, help='number of unlabeled data')
parser.add_argument('--val-iteration', type=int, default=200, help='Number of labeled data')

parser.add_argument('--train_aug', default=False, type=bool, metavar='N', help='aug for training data')

parser.add_argument('--model', type=str, default='bert-base-uncased', help='pretrained model')

parser.add_argument('--data-path', type=str, default='yahoo_answers_csv/', help='path to data folders')

parser.add_argument("--tsa", action='store_true', help="Set this flag if tsa.")
parser.add_argument('--tsa_type', type=str, default='exp', help='tsa type')
parser.add_argument("--lambda-u", default=1.0, type=float, help="lambda_u for consistent loss.")
parser.add_argument("--T", default=1.0, type=float, help="T for sharpening.")
parser.add_argument('--confid', default=0.95, type=float, help='confidence threshold')
parser.add_argument('--confid_prob', default=0.7, type=float, help='confidence threshold for probing words logits')
parser.add_argument("--no_class", default=0, type=int, help="number of class.")

parser.add_argument('--prob_file_name', default='None')
parser.add_argument('--prob_file_name_eval', default='None')
parser.add_argument('--init_best_acc', default=0, type=float, help='yahoo answers:uda(0.6539)')
parser.add_argument('--prob_word_type', default='mixtext', type=str,
                    help="['uda', 'mixtext', 'NEWS', 'CLASS', 'ClassWise', 'LabList', 'SimList', ]")
parser.add_argument('--prob_word_num', default=0, type=int, help='number of probing word')
parser.add_argument('--prob_topk', default=30, type=float, help="selecting percent of probing words from sentence")
parser.add_argument('--parallel_prob', default=False, action="store_true")
parser.add_argument('--wProb', default=1.0, type=float)
parser.add_argument('--prob_loss_func', default='ce', type=str, help='probing words loss function, bce, ce')
parser.add_argument('--add-cls-sep', action="store_true", default=False)
parser.add_argument('--multiple-sep', action="store_true", default=False)
parser.add_argument('--classify-with-cls', action="store_true", default=False)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--seed_l', default=0, type=int)

parser.add_argument('--resume', default="None", type=str, help='resume dir')
parser.add_argument('--specific_name', default=None, type=str, help='the specific name for the output file')
parser.add_argument('--prob_save_sp', default='', type=str, help='the specific name for probing words saving')
parser.add_argument('--eval', action="store_true", default=False)
parser.add_argument('--local_machine', action="store_true", default=False)
args = parser.parse_args()

global_file.init_args()
global_file.init_probing_words_len_list()
global_file.args = args

utils.seed_torch(seed=args.seed)
run_context = utils.RunContext(__file__, args)
writer = SummaryWriter(run_context.tensorboard_dir)
utils.init_logger(run_context)
logger = utils.logger
logger.info('PID:%s', os.getpid())
logger.info('args = %s', args)

from read_data import *
from probing_words_bert1 import ClassificationBertWithProbingWord

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("gpu num: %s", n_gpu)

best_acc = 0
total_steps = 0


def main():
    global best_acc
    train_labeled_set, train_unlabeled_set, val_set, test_set, n_labels, tokenizer = get_data(
        args.data_path, args.n_labeled, train_aug=args.train_aug, add_cls_sep=args.add_cls_sep, mixText_origin=False,
        model=args.model, splited=args.local_machine)
    args.n_labels = n_labels
    labeled_trainloader = Data.DataLoader(dataset=train_labeled_set, batch_size=100, shuffle=False,
                                          drop_last=True)
    unlabeled_trainloader = Data.DataLoader(dataset=train_unlabeled_set, batch_size=args.batch_size_u, shuffle=True,
                                            drop_last=True)
    val_loader = Data.DataLoader(dataset=val_set, batch_size=128, shuffle=False)
    test_loader = Data.DataLoader(dataset=test_set, batch_size=128, shuffle=False)

    cls_token_ids = init_probing_words(tokenizer)

    model = ClassificationBertWithProbingWord(n_labels, model=args.model).cuda()
    model = nn.DataParallel(model)

    model.eval()
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    l_text = train_labeled_set.text
    l_tokens = train_labeled_set.train_tokens
    l_label = train_labeled_set.labels
    with torch.no_grad():
        for batch_idx in range(args.val_iteration):
            try:
                inputs_x, targets_x, inputs_x_length, _, _, idx_x = labeled_train_iter.next()
            except:
                labeled_train_iter = iter(labeled_trainloader)
                inputs_x, targets_x, inputs_x_length, _, _, idx_x = labeled_train_iter.next()

            try:
                (inputs_u, inputs_u2, inputs_ori), (length_u,
                                                    length_u2,
                                                    length_ori), idx_u, targets_u_ori = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                (inputs_u, inputs_u2, inputs_ori), (length_u,
                                                    length_u2,
                                                    length_ori), idx_u, targets_u_ori = unlabeled_train_iter.next()

            # supervised loss
            inputs_x, targets_x, inputs_x_length = inputs_x.cuda(), targets_x.cuda(), inputs_x_length.cuda()

            outputs_all, prob_logits_all, tmp_attention = model(inputs_x, inputs_x_length, output_attention=True,
                                                                probing_words_list=transform_shape(cls_token_ids,
                                                                                                   inputs_x.size()[0]))
            tmp_attention = tmp_attention[-1].detach().cpu()
            ipdb.set_trace()
            # interest_id = 7
            # interest_id = 8
            # interest_id = 19
            # interest_id = 98
            # interest_id = 95
            attended_token_list = list()
            attended_token_len_list = list()
            token_len_list = list()
            len_ratio_list = list()
            id_list = list()
            for interest_id in range(inputs_x.size()[0]):
                cur_token = np.array(l_tokens[interest_id])
                ipdb.set_trace()
                text_probingWords_attention = torch.mean(tmp_attention[interest_id, :, :inputs_x_length[interest_id], -11:-1], 0)
                text_probingWords_attention *= 100
                # text_probingWords_attention /= text_probingWords_attention.sum(1, keepdims=True)
                text_probingWords_attention = torch.softmax(text_probingWords_attention, 1)

                attended_token = cur_token[text_probingWords_attention.argmax(1) == l_label[interest_id]]
                len_ratio = len(attended_token) / len(cur_token)*1.0
                if len(cur_token) < 50 and len_ratio > 0.1:
                    attended_token_list.append(attended_token)
                    attended_token_len_list.append(len(attended_token))
                    token_len_list.append(len(cur_token))
                    len_ratio_list.append(len_ratio)
                    id_list.append(interest_id)



            # inputs_u, inputs_u2, inputs_ori = inputs_u.cuda(), inputs_u2.cuda(), inputs_ori.cuda()
            # length_u, length_u2, length_ori = length_u.cuda(), length_u2.cuda(), length_ori.cuda()
            # targets_u_ori = targets_u_ori.cuda()
            # all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
            # all_lengths = torch.cat([inputs_x_length, length_u, length_u2], dim=0)
            #
            # outputs_all, prob_logits_all, tmp_attention = model(all_inputs, all_lengths, output_attention=True,
            #                                                     probing_words_list=transform_shape(cls_token_ids,
            #                                                                                        all_inputs.size()[0]))
            ipdb.set_trace()



def transform_shape(data, batchsize):
    if args.parallel_prob:
        new_data = list()
        for idx in range(len(data)):
            new_data.append(data[idx].unsqueeze(0).repeat(batchsize, 1))
        return new_data
    else:
        return data


def init_probing_words(tokenizer):
    cls_token_ids = list()
    # cls_wise_attended_tokens = get_class_names(args.data_path)
    cls_wise_attended_tokens = [['society', 'culture'],
                                ['science', 'mathematics'],
                                ['health'],
                                ['education', 'reference'],
                                ['computers', 'internet'],
                                ['football', 'sports','football' ],
                                ['business', 'finance'],
                                ['entertainment', 'music'],
                                ['boyfriend','family',  'relationships'],
                                ['politics', 'government']]
    if args.add_cls_sep and not args.multiple_sep:
        cls_wise_attended_tokens.append(['[SEP]'])
    for id in range(len(cls_wise_attended_tokens)):
        cls_tokens = cls_wise_attended_tokens[id][0]
        cls_token_ids.append(torch.tensor(tokenizer.convert_tokens_to_ids(cls_tokens)).cuda())
    return cls_token_ids


if __name__ == '__main__':
    main()
