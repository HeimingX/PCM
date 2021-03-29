import argparse
import os
import random
from random import shuffle
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
parser.add_argument('--confid_prob', default=0.90, type=float, help='confidence threshold for probing words logits')
parser.add_argument("--no_class", default=0, type=int, help="number of class.")

parser.add_argument('--prob_file_name', default='None')
parser.add_argument('--prob_file_name_eval', default='None')
parser.add_argument('--init_best_acc', default=0, type=float, help='yahoo answers:uda(0.6539)')
parser.add_argument('--prob_word_type', default='mixtext', type=str,
                    help="['uda', 'mixtext', 'NEWS', 'CLASS', 'ClassWise', 'LabList', 'SimList', ]")
parser.add_argument('--prob_word_num', default=0, type=int, help='number of probing word')
parser.add_argument('--prob_word_num_update', default=0, type=int, help='update the probing word num during training')
parser.add_argument('--prob_topk', default=30, type=float, help="selecting percent of probing words from sentence")
parser.add_argument('--parallel_prob', default=False, action="store_true")
parser.add_argument('--wProb', default=1.0, type=float)
parser.add_argument('--prob_loss_func', default='ce', type=str, help='probing words loss function, bce, ce')
parser.add_argument('--add-cls-sep', action="store_true", default=False)
parser.add_argument('--multiple-sep', action="store_true", default=False)
parser.add_argument('--classify-with-cls', action="store_true", default=False)
parser.add_argument('--seed', default=0, type=float)
parser.add_argument('--seed_l', default=0, type=int)

parser.add_argument('--cos_clsifer', action="store_true", default=False)

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
from probing_words_bert import ClassificationBertWithProbingWord

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("gpu num: %s", n_gpu)

best_acc = 0
total_steps = 0
cons_ratio_best = 0


def main():
    global best_acc
    train_labeled_set, train_unlabeled_set, val_set, test_set, n_labels, tokenizer = get_data(
        args.data_path, args.n_labeled, train_aug=args.train_aug, add_cls_sep=args.add_cls_sep, mixText_origin=False,
        model=args.model, splited=args.local_machine)
    args.n_labels = n_labels
    labeled_trainloader = Data.DataLoader(dataset=train_labeled_set, batch_size=args.batch_size, shuffle=True,
                                          drop_last=True)
    unlabeled_trainloader = Data.DataLoader(dataset=train_unlabeled_set, batch_size=args.batch_size_u, shuffle=True,
                                            drop_last=True)
    val_loader = Data.DataLoader(dataset=val_set, batch_size=128, shuffle=False)
    test_loader = Data.DataLoader(dataset=test_set, batch_size=128, shuffle=False)

    cls_token_ids, best_acc = init_probing_words(tokenizer)

    model = ClassificationBertWithProbingWord(n_labels, model=args.model).cuda()

    model = nn.DataParallel(model)
    start_epoch = 0
    ce_criterion = nn.CrossEntropyLoss()
    if args.eval:
        checkpoint_path = args.resume
        assert os.path.isfile(checkpoint_path), "=> no checkpoint found at '{}'".format(checkpoint_path)
        logger.info("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        epoch = checkpoint['epoch']
        val_loss, val_acc = validate(val_loader, model, ce_criterion, epoch, mode='Valid Stats',
                                     cls_token_ids=cls_token_ids)
        logger.info("epoch {}, val acc {}, val_loss {}".format(epoch, val_acc, val_loss))

        test_loss, test_acc = validate(test_loader, model, ce_criterion, epoch, mode='Test Stats ',
                                       cls_token_ids=cls_token_ids)
        logger.info("epoch {}, test acc {},test loss {}".format(epoch, test_acc, test_loss))
        writer.add_scalar('test/loss', test_loss, epoch)
        writer.add_scalar('test/acc', test_acc, epoch)
        ipdb.set_trace()

        # cls_token_ids = update_probing_words(train_labeled_set, train_unlabeled_set, model, n_labels, epoch, tokenizer,
        #                                      probing_words_list=cls_token_ids)
        # ipdb.set_trace()
        return
    elif args.resume and args.resume != "None":
        checkpoint_path = args.resume
        assert os.path.isfile(checkpoint_path), "=> no checkpoint found at '{}'".format(checkpoint_path)
        logger.info("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    optimizer = AdamW(
        [
            {"params": model.module.bert.parameters(), "lr": args.lrmain},
            {"params": model.module.classifier.parameters(), "lr": args.lrlast},
        ])
    train_criterion = SemiLoss(n_labels=n_labels, tsa_type=args.tsa_type)
    test_accs = []
    best_val_acc_loc = 0
    args.val_iteration = len(unlabeled_trainloader)
    for epoch in range(start_epoch, args.epochs):
        ul_pl, ul_pl_logits, ul_pl_match_logits = update_pl(train_unlabeled_set, model, cls_token_ids, epoch)
        ul_pl_stats = (ul_pl, ul_pl_logits, ul_pl_match_logits)
        train(labeled_trainloader, unlabeled_trainloader, model, optimizer, train_criterion, epoch,
              n_labels, args.train_aug, cls_token_ids, ul_pl_stats)
        # save model
        ckpt = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        filename = 'checkpoint.{}.ckpt'.format(run_context.exp_name)
        checkpoint_path = os.path.join(run_context.ckpt_dir, filename)
        torch.save(ckpt, checkpoint_path)

        val_loss, val_acc, val_acc_prob, cons_ratio = validate(val_loader, model, ce_criterion, epoch,
                                                               mode='Valid Stats',
                                                               cls_token_ids=cls_token_ids)
        logger.info(
            "epoch {}, val acc {}, val_loss {}, prob acc {}, consRatio {}".format(epoch, val_acc, val_loss,
                                                                                  val_acc_prob,
                                                                                  cons_ratio))
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)
        writer.add_scalar('val/cons_ratio', cons_ratio, epoch)

        if val_acc >= best_acc:
            best_acc = val_acc
            best_val_acc_loc = epoch
            best_path = os.path.join(run_context.ckpt_dir, 'best.{}.ckpt'.format(run_context.exp_name))
            shutil.copyfile(checkpoint_path, best_path)

        if epoch == 0:
            cons_ratio_best = cons_ratio
        elif cons_ratio > cons_ratio_best:
            cons_ratio_best = cons_ratio
            if args.prob_word_type == 'dynamic':
                # update attended tokens
                cls_token_ids = update_probing_words(train_labeled_set, train_unlabeled_set, model, n_labels, epoch,
                                                     tokenizer, probing_words_list=cls_token_ids)

    logger.info('Best val_acc: {} at {}'.format(best_acc, best_val_acc_loc))
    logger.info('Test acc: {}'.format(test_accs))


def init_probing_words(tokenizer):
    cls_token_ids = list()
    if args.prob_word_type == 'dynamic':
        if args.prob_file_name == 'cls_names':
            if args.eval:
                cls_wise_attended_tokens = np.load(args.data_path + args.prob_file_name_eval, allow_pickle=True)
                cls_wise_attended_tokens = [list(cur_cls_token) for cur_cls_token in cls_wise_attended_tokens]
                logger.info('loaded prob file name: {}'.format(args.prob_file_name))
            else:
                cls_wise_attended_tokens = get_class_names(args.data_path)
                logger.info('loaded prob file name: {}, best eval_acc: {}'.format('class names', 0))
            best_acc = 0
        else:
            cls_wise_attended_tokens = np.load(args.data_path + args.prob_file_name, allow_pickle=True)
            cls_wise_attended_tokens = [list(cur_cls_token) for cur_cls_token in cls_wise_attended_tokens]
            best_acc = args.init_best_acc
            logger.info('loaded prob file name: {}, best eval_acc: {}'.format(args.prob_file_name, best_acc))
    elif args.prob_word_type == 'mixtext' or args.prob_word_type == 'uda' or args.prob_word_type == 'ft':
        cls_wise_attended_tokens = np.load(args.data_path + args.prob_file_name, allow_pickle=True)
        cls_wise_attended_tokens = [list(cur_cls_token) for cur_cls_token in cls_wise_attended_tokens]
        best_acc = 0
        logger.info('loaded prob file name: {}, best eval_acc: {}'.format(args.prob_file_name, best_acc))
    elif args.prob_word_type == 'ClsNames':
        cls_wise_attended_tokens = get_class_names(args.data_path)
        best_acc = 0
        logger.info('loaded prob file name: {}, best eval_acc: {}'.format('class names', 0))
    else:
        raise LookupError

    if args.add_cls_sep and not args.multiple_sep:
        cls_wise_attended_tokens.append(['[SEP]'])
    for id in range(len(cls_wise_attended_tokens)):
        cls_tokens = cls_wise_attended_tokens[id]
        if args.prob_word_num > 0:
            cls_tokens = cls_tokens[:args.prob_word_num]
        if args.eval and args.prob_file_name == 'cls_names' and id < args.n_labels:
            class_names = get_class_names(args.data_path)
            cls_tokens.extend(class_names[id])
        global_file.attended_token_num.append(len(cls_tokens))
        cls_token_ids.append(torch.tensor(tokenizer.convert_tokens_to_ids(cls_tokens)).cuda())
        if args.multiple_sep:
            cls_token_ids.append(torch.tensor(tokenizer.convert_tokens_to_ids(['[SEP]'])).cuda())
    logger.info('number of each class attended tokens: {}'.format(global_file.attended_token_num))
    return cls_token_ids, best_acc


def update_probing_words(train_labeled_set, train_unlabeled_set, model, n_labels, epoch, tokenizer,
                         probing_words_list=None):
    train_labeled_set.update_token_list()
    train_unlabeled_set.update_token_list()
    l_dl = Data.DataLoader(dataset=train_labeled_set, batch_size=args.batch_size, shuffle=False, drop_last=False)
    l_attention, l_y_pred = retrieval_attention.get_attention_all(l_dl, model,
                                                                  data_num=args.n_labeled * n_labels,
                                                                  ul_flag=False, split_tf=False,
                                                                  probing_words_list=probing_words_list,
                                                                  nlabel=n_labels)
    ul_dl = Data.DataLoader(dataset=train_unlabeled_set, batch_size=64, shuffle=False, drop_last=False)
    u_attention, u_y_pred = retrieval_attention.get_attention_all(ul_dl, model,
                                                                  data_num=args.un_labeled * n_labels,
                                                                  ul_flag=True, split_tf=False,
                                                                  probing_words_list=probing_words_list,
                                                                  nlabel=n_labels)
    texts_set = (train_labeled_set.text, train_unlabeled_set.text)
    tokens_set = (train_labeled_set.train_tokens, train_unlabeled_set.train_tokens)
    attention_set = (l_attention, u_attention)
    # y_pred_set = (l_y_pred, u_y_pred)
    y_pred_set = (train_labeled_set.labels, u_y_pred)
    topk = args.prob_topk
    savename = args.data_path + 'uda'
    if args.prob_word_type == 'dynamic':
        savename += '_dyn'
    savename += '_nlab' + str(args.n_labeled)
    savename += '_noClsSep' if not args.add_cls_sep else ''
    savename += '_top' + str(int(topk)) + '_noStopWords_probWrods_ep' + str(epoch) + args.prob_save_sp + '.npy'
    cls_wise_attended_tokens = retrieval_attention.extract_class_wise_attened_tokens(texts_set, tokens_set,
                                                                                     attention_set,
                                                                                     y_pred_set, n_labels, epoch,
                                                                                     savename,
                                                                                     topk, relu_type='probing')

    if args.add_cls_sep and not args.multiple_sep:
        cls_wise_attended_tokens.append(['[SEP]'])
    cls_token_ids = list()
    token_len = list()
    for id in range(len(cls_wise_attended_tokens)):
        cls_tokens = cls_wise_attended_tokens[id]
        if args.prob_word_num > 0:
            if args.prob_word_num_update > args.prob_word_num:
                cls_tokens = cls_tokens[:args.prob_word_num_update]
            else:
                cls_tokens = cls_tokens[:args.prob_word_num]
        if args.prob_file_name == 'cls_names' and id < n_labels:
            class_names = get_class_names(args.data_path)
            cls_tokens.extend(class_names[id])
        token_len.append(len(cls_tokens))
        cls_token_ids.append(torch.tensor(tokenizer.convert_tokens_to_ids(cls_tokens)).cuda())
        if args.multiple_sep:
            cls_token_ids.append(torch.tensor(tokenizer.convert_tokens_to_ids(['[SEP]'])).cuda())
    logger.info('number of each class attended tokens: {}'.format(token_len))
    return cls_token_ids


def transform_shape(data, batchsize):
    if args.parallel_prob:
        new_data = list()
        for idx in range(len(data)):
            new_data.append(data[idx].unsqueeze(0).repeat(batchsize, 1))
        return new_data
    else:
        return data


def update_pl(dataset, model, cls_token_ids, epoch):
    model.eval()
    num_ul = args.n_labels * args.un_labeled
    ul_gt = torch.ones(num_ul).long().cuda() * -1
    ul_pl = torch.ones(num_ul).long().cuda() * -1
    ul_pl_logits = torch.zeros((num_ul, args.n_labels)).cuda()
    ul_pl_match_logits = torch.zeros((num_ul, args.n_labels)).cuda()
    pos_idx = list()
    neg_idx = list()
    dl = Data.DataLoader(dataset=dataset, batch_size=128, shuffle=False, drop_last=False)
    with torch.no_grad():
        for batch_idx, ((_, _, inputs_ori), (_, _, length_ori), idx_u, targets_u_ori) in enumerate(dl):
            inputs_ori = inputs_ori.cuda()
            length_ori = length_ori.cuda()
            targets_u_ori = targets_u_ori.cuda()

            ul_gt[idx_u] = targets_u_ori
            orig_y_pred, prob_logits_orig = model(inputs_ori, length_ori, use_cls_sep=args.add_cls_sep,
                                                  probing_words_list=transform_shape(cls_token_ids,
                                                                                     inputs_ori.size()[0]))

            orig_y_pred = orig_y_pred / args.T
            ul_pl_logits[idx_u] = orig_y_pred
            ul_pl_match_logits[idx_u] = prob_logits_orig

            # generate pseudo-label mask: cond1
            targets_u = torch.softmax(orig_y_pred, dim=1)
            pl_u, pl_u_idx = torch.max(targets_u, 1)
            loss_u_mask = (pl_u >= args.confid).float()
            # cond2
            if args.prob_loss_func == 'bce':
                prob_proba_orig = torch.sigmoid(prob_logits_orig)
                prob_mask = ((prob_proba_orig > args.confid_prob).float().sum(1) == 1).float()
            else:
                prob_proba_orig = torch.softmax(prob_logits_orig, dim=1)
                max_prob_proba = torch.max(prob_proba_orig, 1)[0]
                prob_mask = (max_prob_proba > args.confid_prob).float()
            loss_u_mask *= prob_mask
            # cond3
            pl_u_match, pl_u_idx_match = torch.max(prob_proba_orig, 1)
            same_vote_mask = (pl_u_idx == pl_u_idx_match).float()
            loss_u_mask *= same_vote_mask

            # get pos/neg pl idx
            pos_mask = (pl_u_idx == 1) * loss_u_mask
            pos_mask = pos_mask >= 1
            pos_idx.extend(idx_u[pos_mask].tolist())
            neg_mask = (pl_u_idx == 0) * loss_u_mask
            neg_mask = neg_mask >= 1
            neg_idx.extend(idx_u[neg_mask].tolist())

            # get satisfied pl
            loss_u_mask = loss_u_mask >= 1
            _pl = torch.ones(inputs_ori.size()[0]).long().cuda() * -1
            _pl[loss_u_mask] = pl_u_idx[loss_u_mask]
            ul_pl[idx_u] = _pl
            del _pl

        # balance the pos/neg pl
        pos_pl_num = len(pos_idx)
        neg_pl_num = len(neg_idx)
        if pos_pl_num > neg_pl_num:
            shuffle(pos_idx)
            ul_pl[pos_idx[neg_pl_num:]] = -1
            final_pl_num = 2 * neg_pl_num
        elif pos_pl_num < neg_pl_num:
            shuffle(neg_idx)
            ul_pl[neg_idx[pos_pl_num:]] = -1
            final_pl_num = 2 * pos_pl_num
        else:
            final_pl_num = pos_pl_num + neg_pl_num

        mask_ratio = final_pl_num / num_ul
        writer.add_scalar('monitor/mask', mask_ratio, epoch)
        if final_pl_num > 0:
            mask_acc = (ul_pl == ul_gt).float().sum() / final_pl_num
            writer.add_scalar('monitor/mask_acc', mask_acc, epoch)
        else:
            mask_acc = 0
        logger.info('epoch{}, mask {} - {}'.format(epoch, mask_ratio, mask_acc))
    return ul_pl, ul_pl_logits, ul_pl_match_logits


def validate(valloader, model, criterion, epoch, mode, cls_token_ids=None):
    model.eval()

    data_time = 0
    train_time = 0
    end = time.time()
    with torch.no_grad():
        loss_total = 0
        total_sample = 0
        correct = 0
        correct_prob = 0

        _total_sample = args.n_labels * 2000
        pred_pl = torch.zeros(_total_sample).long()
        prob_pl = torch.zeros(_total_sample, args.n_labels)
        match_pl = torch.zeros(_total_sample, args.n_labels)

        for batch_idx, (inputs, targets, length, _, _, idx) in enumerate(valloader):
            data_time += time.time() - end
            end = time.time()
            inputs, targets, length = inputs.cuda(), targets.cuda(non_blocking=True), length.cuda()
            if cls_token_ids is None:
                outputs = model(inputs, length, probing_words_list=cls_token_ids, use_cls_sep=args.add_cls_sep)
            else:
                outputs, outputs_prob = model(inputs, length,
                                              probing_words_list=transform_shape(cls_token_ids, inputs.size()[0]),
                                              use_cls_sep=args.add_cls_sep)
                predicted_prob = torch.max(outputs_prob.data, 1)[1]
                correct_prob += (np.array(predicted_prob.cpu()) == np.array(targets.cpu())).sum()
                match_pl[idx] = outputs_prob.cpu()
            loss = criterion(outputs, targets)

            _, predicted = torch.max(outputs.data, 1)
            correct += (np.array(predicted.cpu()) == np.array(targets.cpu())).sum()
            loss_total += loss.item() * inputs.shape[0]
            total_sample += inputs.shape[0]

            pred_pl[idx] = predicted.cpu()
            prob_pl[idx] = outputs.cpu()

            train_time += time.time() - end
            end = time.time()
    logger.info('test time: data {}, infer {}'.format(data_time, train_time))

    acc_total = correct / total_sample
    loss_total = loss_total / total_sample
    acc_prob = correct_prob / total_sample

    # cond1:
    prob_pl = prob_pl / args.T
    prob_pl = torch.softmax(prob_pl, dim=1)
    pl_u, pl_u_idx = torch.max(prob_pl, 1)
    # sem_mask = (pl_u >= args.confid).float()
    # cond2:
    match_pl = torch.sigmoid(match_pl)
    # match_mask = ((match_pl > args.confid_prob).float().sum(1) == 1).float()
    # cond3:
    match_idx = torch.max(match_pl, 1)[1]
    same_vote_mask = (pl_u_idx == match_idx).float()

    whole_mask = same_vote_mask
    cons_ratio = sum(whole_mask) / total_sample
    return loss_total, acc_total, acc_prob, cons_ratio


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, criterion, epoch, n_labels, train_aug,
          cls_token_ids, ul_pl_stats):
    model.train()
    global total_steps

    ul_pl, ul_pl_logits, ul_pl_match_logits = ul_pl_stats
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    data_time = 0
    train_time = 0
    end = time.time()
    for batch_idx in range(args.val_iteration):
        total_steps += 1

        if not train_aug:
            try:
                inputs_x, targets_x, inputs_x_length, _, _, idx_x = labeled_train_iter.next()
            except:
                labeled_train_iter = iter(labeled_trainloader)
                inputs_x, targets_x, inputs_x_length, _, _, idx_x = labeled_train_iter.next()
        else:
            try:
                (inputs_x, inputs_x_aug), (targets_x, _), (inputs_x_length,
                                                           inputs_x_length_aug) = labeled_train_iter.next()
            except:
                labeled_train_iter = iter(labeled_trainloader)
                (inputs_x, inputs_x_aug), (targets_x, _), (inputs_x_length,
                                                           inputs_x_length_aug) = labeled_train_iter.next()
        try:
            (inputs_u, inputs_u2, inputs_ori), (length_u,
                                                length_u2,
                                                length_ori), idx_u, targets_u_ori = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2, inputs_ori), (length_u,
                                                length_u2,
                                                length_ori), idx_u, targets_u_ori = unlabeled_train_iter.next()
        data_time += time.time() - end
        end = time.time()

        # supervised loss
        inputs_x, targets_x, inputs_x_length = inputs_x.cuda(), targets_x.cuda(), inputs_x_length.cuda()
        inputs_u, inputs_u2, inputs_ori = inputs_u.cuda(), inputs_u2.cuda(), inputs_ori.cuda()
        length_u, length_u2, length_ori = length_u.cuda(), length_u2.cuda(), length_ori.cuda()

        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_lengths = torch.cat([inputs_x_length, length_u, length_u2], dim=0)
        outputs_all, prob_logits_all = model(all_inputs, all_lengths, use_cls_sep=args.add_cls_sep,
                                             probing_words_list=transform_shape(cls_token_ids, all_inputs.size()[0]))
        outputs_x = outputs_all[:args.batch_size]
        aug_y_pred = outputs_all[args.batch_size:args.batch_size + args.batch_size_u]
        aug_y_pred2 = outputs_all[-args.batch_size_u:]
        prob_logits_x = prob_logits_all[:args.batch_size]
        prob_logits_aug = prob_logits_all[args.batch_size:args.batch_size + args.batch_size_u]
        prob_logits_aug2 = prob_logits_all[-args.batch_size_u:]

        # outputs_x, prob_logits_x = model(inputs_x, inputs_x_length, use_cls_sep=args.add_cls_sep,
        #                                  probing_words_list=transform_shape(cls_token_ids, inputs_x.size()[0]))
        # # uda loss
        # aug_y_pred, prob_logits_aug = model(inputs_u, length_u, use_cls_sep=args.add_cls_sep,
        #                                     probing_words_list=transform_shape(cls_token_ids, inputs_u.size()[0]))
        # aug_y_pred2, prob_logits_aug2 = model(inputs_u2, length_u2, use_cls_sep=args.add_cls_sep,
        #                                       probing_words_list=transform_shape(cls_token_ids, inputs_u2.size()[0]))
        orig_y_pred = ul_pl_logits[idx_u]
        prob_logits_orig = ul_pl_match_logits[idx_u]
        loss_u_mask = (ul_pl[idx_u] > -1).float()

        loss = criterion(outputs_x=outputs_x, targets_x=targets_x, prob_logits_x=prob_logits_x,
                         outputs_u=(aug_y_pred, aug_y_pred2), targets_u=(orig_y_pred, prob_logits_orig),
                         prob_logits_u=(prob_logits_aug, prob_logits_aug2), loss_u_mask=loss_u_mask,
                         epoch=epoch + batch_idx / args.val_iteration, loss_type=args.prob_loss_func)

        writer.add_scalar('train/loss_total', loss[0], batch_idx + args.val_iteration * epoch)
        writer.add_scalar('train/loss_l', loss[1], batch_idx + args.val_iteration * epoch)
        writer.add_scalar('train/loss_uda', loss[2], batch_idx + args.val_iteration * epoch)
        writer.add_scalar('train/loss_l_prob', loss[3], batch_idx + args.val_iteration * epoch)
        writer.add_scalar('train/loss_u_prob', loss[4], batch_idx + args.val_iteration * epoch)
        if batch_idx % 100 == 0:
            logger.info(
                'epoch {}-{}, loss: total {}, l {}, ul {}, lprob {}, uprob {}'.format(epoch, batch_idx, loss[0],
                                                                                      loss[1], loss[2], loss[3],
                                                                                      loss[4]))

        total_loss = loss[0]
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        train_time += time.time() - end
        end = time.time()
    logger.info('train time: data {}, infer {}'.format(data_time, train_time))
    return labeled_train_iter, unlabeled_train_iter


class SemiLoss(object):
    def __init__(self, n_labels=2, tsa_type='exp'):
        self.n_labels = n_labels
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.bce_loss = nn.BCELoss(reduction='none')
        self.sigmoid = nn.Sigmoid()
        self.tsa_type = tsa_type

    @staticmethod
    def TSA(epoch, n_class, tsa_type='exp'):
        epoch = math.floor(epoch) / args.epochs
        if tsa_type == 'exp':
            return np.exp((epoch - 1) * 5) * (1 - 1 / n_class) + 1 / n_class
        elif tsa_type == 'linear':
            return epoch * (1 - 1 / n_class) + 1 / n_class
        elif tsa_type == 'log':
            return (1 - np.exp(-epoch * 5)) * (1 - 1 / n_class) + 1 / n_class
        else:
            return 1

    @staticmethod
    def linear_rampup(current, rampup_length=args.epochs):
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current / rampup_length, 0.0, 1.0)
            return float(current)

    @staticmethod
    def kl_divergence_from_logits(p_logits, q_logits):
        p = torch.softmax(p_logits, dim=1)
        log_p = torch.log_softmax(p_logits, dim=1)
        log_q = torch.log_softmax(q_logits, dim=1)
        kl = (p * (log_p - log_q)).sum(-1)
        return kl

    @staticmethod
    def oneVSall_loss(logits, labels, salt=1e-24):
        bin_logits = torch.sigmoid(logits)
        loss = -1 * (F.logsigmoid(logits) * labels).sum(1)
        loss += -1 * (torch.log(1 - bin_logits + salt) * (1 - labels)).sum(1)
        return loss

    def __call__(self, outputs_x, targets_x, prob_logits_x, outputs_u=None, targets_u=None, prob_logits_u=None,
                 loss_u_mask=None, epoch=None, loss_type='bce'):
        loss = self.ce_loss(outputs_x.view(-1, self.n_labels), targets_x.view(-1))
        if args.tsa:
            thres = self.TSA(epoch, self.n_labels, self.tsa_type)
            q_y_softmax = F.softmax(outputs_x, dim=1)
            targets_x_1hot = torch.zeros(outputs_x.shape[0], self.n_labels).scatter_(1, targets_x.cpu().view(-1, 1),
                                                                                     1).cuda()
            q_correct = torch.sum(targets_x_1hot * q_y_softmax, dim=-1)
            loss_l_mask = (q_correct <= thres).float().detach()  # Ignore confident predictions.
            loss = (loss * loss_l_mask).sum() / max(1, loss_l_mask.sum())
        else:
            loss = loss.mean()

        targets_x_1hot = torch.zeros(outputs_x.shape[0], self.n_labels).scatter_(1, targets_x.cpu().view(-1, 1), 1)
        if loss_type == '1-vs-all':
            loss_prob_x = self.oneVSall_loss(prob_logits_x.view(-1, self.n_labels), targets_x_1hot.cuda()).mean()
        elif loss_type == 'bce':
            loss_prob_x = self.bce_loss(self.sigmoid(prob_logits_x.view(-1)), targets_x_1hot.view(-1).cuda()).mean()
            loss_prob_x *= args.wProb
        else:
            loss_prob_x = self.ce_loss(prob_logits_x.view(-1, self.n_labels), targets_x.view(-1)).mean()
        # loss_prob_x = 0

        # ul loss
        if loss_u_mask.sum() > 0:
            outputs_u1, outputs_u2 = outputs_u
            targets_u_pred, targets_u_prob = targets_u
            loss_u1 = (self.kl_divergence_from_logits(targets_u_pred,
                                                      outputs_u1) * loss_u_mask).sum() / loss_u_mask.sum()
            loss_u2 = (self.kl_divergence_from_logits(targets_u_pred,
                                                      outputs_u2) * loss_u_mask).sum() / loss_u_mask.sum()
            loss_uda = (loss_u1 + loss_u2) / 2 * args.lambda_u  # * self.linear_rampup(epoch)

            # ipdb.set_trace()
            prob_logits_u1, prob_logits_u2 = prob_logits_u
            if loss_type == '1-vs-all':
                # ============ 1-vs-all loss ============
                loss_prob_u1 = self.oneVSall_loss(prob_logits_u1.view(-1, self.n_labels),
                                                  torch.softmax(targets_u, dim=1))
                loss_prob_u1 = (loss_prob_u1 * loss_u_mask).sum() / loss_u_mask.sum()
                loss_prob_u2 = self.oneVSall_loss(prob_logits_u2.view(-1, self.n_labels),
                                                  torch.softmax(targets_u, dim=1))
                loss_prob_u2 = (loss_prob_u2 * loss_u_mask).sum() / loss_u_mask.sum()
            elif loss_type == 'bce':
                # ============ bce loss ============
                bs_u, num_cls = outputs_u1.shape
                loss_prob_mask = loss_u_mask.unsqueeze(1).expand(bs_u, num_cls).reshape(-1)
                loss_prob_u1 = self.bce_loss(self.sigmoid(prob_logits_u1.view(-1)),
                                             torch.sigmoid(targets_u_prob).view(-1))
                loss_prob_u1 = (loss_prob_u1 * loss_prob_mask).sum() / loss_prob_mask.sum()
                loss_prob_u2 = self.bce_loss(self.sigmoid(prob_logits_u2.view(-1)),
                                             torch.sigmoid(targets_u_prob).view(-1))
                loss_prob_u2 = (loss_prob_u2 * loss_prob_mask).sum() / loss_prob_mask.sum()
            else:
                # ============ cross-entropy loss ============
                # loss_prob_u1 = self.ce_loss(prob_logits_u1.view(-1, self.n_labels), torch.max(targets_u_prob, 1)[1])
                # loss_prob_u1 = (loss_prob_u1 * loss_u_mask).sum() / loss_u_mask.sum()
                # loss_prob_u2 = self.ce_loss(prob_logits_u2.view(-1, self.n_labels), torch.max(targets_u_prob, 1)[1])
                # loss_prob_u2 = (loss_prob_u2 * loss_u_mask).sum() / loss_u_mask.sum()
                loss_prob_u1 = (self.kl_divergence_from_logits(targets_u_prob,
                                                               prob_logits_u1) * loss_u_mask).sum() / loss_u_mask.sum()
                loss_prob_u2 = (self.kl_divergence_from_logits(targets_u_prob,
                                                               prob_logits_u2) * loss_u_mask).sum() / loss_u_mask.sum()
            loss_prob = (loss_prob_u1 + loss_prob_u2) / 2 * args.wProb
            # loss_prob = 0
        else:
            loss_uda, loss_prob = 0, 0

        total_loss = loss + loss_uda + (loss_prob_x + loss_prob)
        return total_loss, loss, loss_uda, loss_prob_x, loss_prob


if __name__ == '__main__':
    main()
