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

parser = argparse.ArgumentParser(description='PyTorch UDA Models')

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
parser.add_argument("--no_class", default=0, type=int, help="number of class.")
parser.add_argument('--prob_topk', default=30, type=float, help="selecting percent of probing words from sentence")
parser.add_argument('--add-cls-sep', action="store_true", default=False)
parser.add_argument('--classify-with-cls', action="store_true", default=False)
parser.add_argument('--seed', default=0, type=float)
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
from normal_bert import ClassificationBert

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
    labeled_trainloader = Data.DataLoader(dataset=train_labeled_set, batch_size=args.batch_size, shuffle=True,
                                          drop_last=True)
    unlabeled_trainloader = Data.DataLoader(dataset=train_unlabeled_set, batch_size=args.batch_size_u, shuffle=True,
                                            drop_last=True)
    val_loader = Data.DataLoader(dataset=val_set, batch_size=128, shuffle=False)
    test_loader = Data.DataLoader(dataset=test_set, batch_size=128, shuffle=False)

    model = ClassificationBert(n_labels, use_gap=not args.classify_with_cls, require_attention=True,
                               use_cls_sep=args.add_cls_sep).cuda()
    # model = ClassificationBert(n_labels, use_gap=not args.classify_with_cls, require_attention=False, use_cls_sep=args.add_cls_sep).cuda()

    model = nn.DataParallel(model)
    optimizer = AdamW(
        [
            {"params": model.module.bert.parameters(), "lr": args.lrmain},
            {"params": model.module.classifier.parameters(), "lr": args.lrlast},
        ])

    ce_criterion = nn.CrossEntropyLoss()
    train_criterion = SemiLoss(n_labels=n_labels, tsa_type=args.tsa_type)

    if args.eval:
        checkpoint_path = args.resume
        assert os.path.isfile(checkpoint_path), "=> no checkpoint found at '{}'".format(checkpoint_path)
        logger.info("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        epoch = checkpoint['epoch']
        # ====== get acc on val/test set ======
        val_loss, val_acc = validate(val_loader, model, ce_criterion, epoch, mode='Valid Stats')
        logger.info("epoch {}, val acc {}, val_loss {}".format(epoch, val_acc, val_loss))
        test_loss, test_acc = validate(test_loader, model, ce_criterion, epoch, mode='Test Stats ')
        logger.info("epoch {}, test acc {},test loss {}".format(epoch, test_acc, test_loss))
        writer.add_scalar('test/loss', test_loss, epoch)
        writer.add_scalar('test/acc', test_acc, epoch)
        ipdb.set_trace()

        # ====== get attended tokens ======
        l_dl = Data.DataLoader(dataset=train_labeled_set, batch_size=args.batch_size, shuffle=False, drop_last=False)
        l_attention, l_y_pred = retrieval_attention.get_attention_all(l_dl, model, data_num=args.n_labeled * n_labels,
                                                                      ul_flag=False, split_tf=False, nlabel=n_labels)
        ul_dl = Data.DataLoader(dataset=train_unlabeled_set, batch_size=128, shuffle=False, drop_last=False)
        u_attention, u_y_pred = retrieval_attention.get_attention_all(ul_dl, model, data_num=args.un_labeled * n_labels,
                                                                      ul_flag=True, split_tf=False, nlabel=n_labels)
        texts_set = (train_labeled_set.text, train_unlabeled_set.text)
        tokens_set = (train_labeled_set.train_tokens, train_unlabeled_set.train_tokens)
        attention_set = (l_attention, u_attention)
        # y_pred_set = (l_y_pred, u_y_pred)
        y_pred_set = (train_labeled_set.labels, u_y_pred)
        topk = args.prob_topk
        savename = args.data_path + 'uda'
        savename += '_nlab' + str(args.n_labeled)
        savename += '_noClsSep' if not args.add_cls_sep else ''
        savename += '_top' + str(int(topk)) + '_noStopWords_probWrods_ep' + str(epoch) + '.npy'
        attended_set = retrieval_attention.extract_class_wise_attened_tokens(texts_set, tokens_set, attention_set,
                                                                             y_pred_set, n_labels, epoch, savename,
                                                                             topk)
        attened_set_len = [len(attended_set[i]) for i in range(len(attended_set))]
        print('attended_set len: {}'.format(attened_set_len))
        ipdb.set_trace()
        return

    test_accs = []
    best_val_acc_loc = 0
    for epoch in range(args.epochs):
        train(labeled_trainloader, unlabeled_trainloader, model, optimizer, train_criterion, epoch, n_labels,
              args.train_aug)
        # save model
        ckpt = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        filename = 'checkpoint.{}.ckpt'.format(run_context.exp_name)
        checkpoint_path = os.path.join(run_context.ckpt_dir, filename)
        torch.save(ckpt, checkpoint_path)

        val_loss, val_acc = validate(val_loader, model, ce_criterion, epoch, mode='Valid Stats')
        logger.info("epoch {}, val acc {}, val_loss {}".format(epoch, val_acc, val_loss))
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)

        if val_acc >= best_acc:
            best_acc = val_acc
            best_val_acc_loc = epoch
            best_path = os.path.join(run_context.ckpt_dir, 'best.{}.ckpt'.format(run_context.exp_name))
            shutil.copyfile(checkpoint_path, best_path)

    logger.info('Best val_acc: {} at {}'.format(best_acc, best_val_acc_loc))
    logger.info('Test acc: {}'.format(test_accs))


def validate(valloader, model, criterion, epoch, mode):
    model.eval()

    data_time = 0
    train_time = 0
    end = time.time()
    with torch.no_grad():
        loss_total = 0
        total_sample = 0
        acc_total = 0
        correct = 0

        false_id_set = list()
        false_prediction_set = list()
        prediction_set = list()
        target_set = list()
        cam_set = list()
        for batch_idx, (inputs, targets, length, _, _, idx) in enumerate(valloader):
            data_time += time.time() - end
            end = time.time()
            inputs, targets, length = inputs.cuda(), targets.cuda(non_blocking=True), length.cuda()
            outputs = model(inputs, length)
            loss = criterion(outputs, targets)

            _, predicted = torch.max(outputs.data, 1)
            correct += (np.array(predicted.cpu()) == np.array(targets.cpu())).sum()
            loss_total += loss.item() * inputs.shape[0]
            total_sample += inputs.shape[0]

            # correct_prediction = np.array(predicted.cpu()) == np.array(targets.cpu())
            # false_id_set.append(idx[~correct_prediction])
            # pred_prob = torch.softmax(outputs.data, 1)
            # false_prediction_set.append(pred_prob[~correct_prediction])
            # prediction_set.append(pred_prob)
            # target_set.append(targets)

            # output_value, _ = torch.max(outputs, 1)
            # output_value.backward()
            # gradients = model.get_activations_gradient().detach()
            # pooled_gradients = torch.mean(gradients, dim=[0, 1], keepdim=True)
            # activations = model.get_activations().detach()
            # bs, seq_len, chn = activations.shape
            # weighted_act = pooled_gradients.repeat(bs, seq_len, 1) * activations
            # heatmap = torch.mean(weighted_act, dim=2).squeeze()
            # cam_set.append(heatmap)
            # model.zero_grad()

            train_time += time.time() - end
            end = time.time()
    logger.info('test time: data {}, infer {}'.format(data_time, train_time))

    acc_total = correct / total_sample
    loss_total = loss_total / total_sample

    # f_p = torch.cat(false_prediction_set, 0)
    # f_id = torch.cat(false_id_set)
    # all_p = torch.cat(prediction_set, 0)
    # all_l = torch.cat(target_set)
    # all_cam = torch.cat(cam_set, 0)

    return loss_total, acc_total  # , f_p, f_id, all_p, all_l, all_cam


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, criterion, epoch, n_labels, train_aug):
    model.train()
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    global total_steps
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
        targets_u_ori = targets_u_ori.cuda()

        # outputs_x = model(inputs_x, inputs_x_length)
        # aug_y_pred = model(inputs_u, length_u)
        # aug_y_pred2 = model(inputs_u2, length_u2)

        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_lengths = torch.cat([inputs_x_length, length_u, length_u2], dim=0)
        outputs_all = model(all_inputs, all_lengths)
        outputs_x = outputs_all[:args.batch_size]
        aug_y_pred = outputs_all[args.batch_size:args.batch_size + args.batch_size_u]
        aug_y_pred2 = outputs_all[-args.batch_size_u:]

        with torch.no_grad():
            orig_y_pred = model(inputs_ori, length_ori)
            orig_y_pred = orig_y_pred / args.T
            targets_u = torch.softmax(orig_y_pred, dim=1)
            pl_u, pl_u_idx = torch.max(targets_u, 1)
            loss_u_mask = (pl_u >= args.confid).float()
            writer.add_scalar('monitor/mask', loss_u_mask.mean(), batch_idx + args.val_iteration * epoch)
            pl_pred = orig_y_pred.argmax(1)
            pl_acc = (pl_pred == targets_u_ori).float()
            writer.add_scalar('monitor/pl_acc', pl_acc.mean(), batch_idx + args.val_iteration * epoch)

            # verified wity ground truth
            # loss_u_mask *= (pl_u_idx == targets_u_ori).float()

            if loss_u_mask.sum() > 0:
                pl_acc = (pl_acc * loss_u_mask).sum() / loss_u_mask.sum()
            else:
                pl_acc = 0
            writer.add_scalar('monitor/mask_acc', pl_acc, batch_idx + args.val_iteration * epoch)

        loss = criterion(outputs_x, targets_x, (aug_y_pred, aug_y_pred2), orig_y_pred, loss_u_mask,
                         epoch + batch_idx / args.val_iteration)
        writer.add_scalar('train/loss_total', loss[0].item(), batch_idx + args.val_iteration * epoch)
        writer.add_scalar('train/loss_l', loss[1].item(), batch_idx + args.val_iteration * epoch)
        writer.add_scalar('train/loss_uda', loss[2].item(), batch_idx + args.val_iteration * epoch)
        if batch_idx % 100 == 0:
            logger.info(
                'epoch {}-{}, loss: total {}, l {}, ul {}, mask {} - {}'.format(epoch, batch_idx, loss[0], loss[1],
                                                                                loss[2], loss_u_mask.mean(), pl_acc))

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
        self.loss_fct = nn.CrossEntropyLoss(reduction='none')
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
    def loss_label_smoothing(outputs, labels):
        """
        loss function for label smoothing regularization
        """
        alpha = 0.1
        N = outputs.size(0)  # batch_size
        C = outputs.size(1)  # number of classes
        smoothed_labels = torch.full(size=(N, C), fill_value=alpha / (C - 1)).cuda()
        smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=1 - alpha)

        log_prob = torch.nn.functional.log_softmax(outputs, dim=1)
        loss = -torch.sum(log_prob * smoothed_labels) / N

        return loss

    def __call__(self, outputs_x, targets_x, outputs_u=None, targets_u=None, loss_u_mask=None, epoch=None,
                 label_smooth=False):
        if label_smooth:
            loss = self.loss_label_smoothing(outputs_x.view(-1, self.n_labels), targets_x.view(-1))
        else:
            loss = self.loss_fct(outputs_x.view(-1, self.n_labels), targets_x.view(-1))

            if args.tsa:
                thres = self.TSA(epoch, self.n_labels, self.tsa_type)
                q_y_softmax = F.softmax(outputs_x, dim=1)
                targets_x_1hot = torch.zeros(outputs_x.shape[0], self.n_labels).scatter_(1, targets_x.cpu().view(-1, 1),
                                                                                         1)
                q_correct = torch.sum(targets_x_1hot.cuda() * q_y_softmax, dim=-1)
                loss_l_mask = (q_correct <= thres).float().detach()  # Ignore confident predictions.
                loss = (loss * loss_l_mask).sum() / max(1, loss_l_mask.sum())
            else:
                loss = loss.mean()

        outputs_u1, outputs_u2 = outputs_u
        loss_u1 = (self.kl_divergence_from_logits(targets_u, outputs_u1) * loss_u_mask).mean()
        loss_u2 = (self.kl_divergence_from_logits(targets_u, outputs_u2) * loss_u_mask).mean()
        loss_uda = (loss_u1 + loss_u2) / 2 * args.lambda_u  # * self.linear_rampup(epoch)

        total_loss = loss + loss_uda
        return total_loss, loss, loss_uda


if __name__ == '__main__':
    main()
