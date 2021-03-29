import argparse
import os
import random
import math
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from pytorch_transformers import *
from torch.autograd import Variable
from torch.utils.data import Dataset
from tensorboardX import SummaryWriter

import utils
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
parser.add_argument("--no_class", default=0, type=int, help="number of class.")

parser.add_argument('--prob_word_type', default='CLASS', type=str,
                    help="['NEWS', 'CLASS', 'ClassWise', 'LabList', 'SimList']")
parser.add_argument('--add-cls-sep', action="store_true", default=False)
parser.add_argument('--classify-with-cls', action="store_true", default=False)
parser.add_argument('--split-match', action="store_true", default=False)

parser.add_argument('--resume', default="None", type=str, help='resume dir')
parser.add_argument('--specific_name', default=None, type=str, help='the specific name for the output file')
parser.add_argument('--eval', action="store_true", default=False)
args = parser.parse_args()

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


def main():
    global best_acc
    train_labeled_set, train_unlabeled_set, val_set, test_set, n_labels, tokenizer = get_data(
        args.data_path, args.n_labeled, train_aug=args.train_aug, add_cls_sep=args.add_cls_sep, mixText_origin=False,
        model=args.model)
    labeled_trainloader = Data.DataLoader(dataset=train_labeled_set, batch_size=args.batch_size, shuffle=True,
                                          drop_last=True)
    unlabeled_trainloader = Data.DataLoader(dataset=train_unlabeled_set, batch_size=args.batch_size_u, shuffle=True,
                                            drop_last=True)
    val_loader = Data.DataLoader(dataset=val_set, batch_size=256, shuffle=False)
    test_loader = Data.DataLoader(dataset=test_set, batch_size=256, shuffle=False)

    cls_token_ids = list()
    attended_token_num = list()
    if args.add_cls_sep:
        # here append multiple [SEP] to meet the dataparallel's requirement
        CLS_WISE_ATTENTED_TOKENS.append(['[SEP]'] * n_gpu)
    for cls_tokens in CLS_WISE_ATTENTED_TOKENS:
        attended_token_num.append(len(cls_tokens))
        cls_token_ids.append(torch.tensor(tokenizer.convert_tokens_to_ids(cls_tokens)).cuda())
    logger.info('number of each class attended tokens: {}'.format(attended_token_num))

    model = ClassificationBertWithProbingWord(n_labels, model=args.model).cuda()
    # if args.resume and args.resume != "None":
    #     resume_filename = args.resume
    #     assert os.path.isfile(resume_filename), "=> no checkpoint found at '{}'".format(resume_filename)
    #     logger.info("=> loading checkpoint '{}'".format(resume_filename))
    #     checkpoint = torch.load(resume_filename)
    #     resume_dict = checkpoint['state_dict']
    #     model_dict = model.state_dict()
    #     filter_dict = {k: v for k, v in resume_dict.items() if k in model_dict}
    #     model_dict.update(filter_dict)
    #     model.load_state_dict(model_dict)

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

        # if isinstance(model, torch.nn.DataParallel):
        #     model = model.module
        # import ipdb; ipdb.set_trace()
        # test_res = np.load('./yahoo_test_cam.npz')
        # test_loss, test_acc, f_p, f_id, all_p, all_l, all_cam = validate(test_loader, model, criterion, epoch, mode='Test Stats ')
        # cls_token_ids = None
        val_loss, val_acc = validate(val_loader, model, ce_criterion, epoch, mode='Valid Stats',
                                     cls_token_ids=cls_token_ids)
        logger.info("epoch {}, val acc {}, val_loss {}".format(epoch, val_acc, val_loss))

        test_loss, test_acc = validate(test_loader, model, ce_criterion, epoch, mode='Test Stats ',
                                       cls_token_ids=cls_token_ids)
        logger.info("epoch {}, test acc {},test loss {}".format(epoch, test_acc, test_loss))
        # writer.add_scalar('test/loss', test_loss, epoch)
        # writer.add_scalar('test/acc', test_acc, epoch)
        return

    test_accs = []
    best_val_acc_loc = 0
    labeled_train_iter, unlabeled_train_iter = None, None
    for epoch in range(args.epochs):
        labeled_train_iter, unlabeled_train_iter = train(labeled_trainloader, labeled_train_iter, unlabeled_trainloader,
                                                         unlabeled_train_iter, model, optimizer, train_criterion, epoch,
                                                         n_labels, args.train_aug, cls_token_ids)
        # save model
        ckpt = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        filename = 'checkpoint.{}_ep{}.ckpt'.format(run_context.exp_name, epoch)
        ckpt_root_path = './ckpt/udaPrbWrd'
        checkpoint_path = os.path.join(ckpt_root_path, filename)
        os.makedirs(ckpt_root_path, exist_ok=True)
        torch.save(ckpt, checkpoint_path)

        # assert os.path.isfile(checkpoint_path), "=> no checkpoint found at '{}'".format(checkpoint_path)
        # logger.info("=> loading checkpoint '{}'".format(checkpoint_path))
        # checkpoint = torch.load(checkpoint_path)
        # model.load_state_dict(checkpoint['state_dict'])

        val_loss, val_acc = validate(val_loader, model, ce_criterion, epoch, mode='Valid Stats',
                                     cls_token_ids=cls_token_ids)
        logger.info("epoch {}, val acc {}, val_loss {}".format(epoch, val_acc, val_loss))
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)

        if val_acc >= best_acc:
            best_acc = val_acc
            best_val_acc_loc = epoch
            # test_loss, test_acc = validate(test_loader, model, criterion, epoch, mode='Test Stats ')
            # test_accs.append(test_acc)
            # logger.info("epoch {}, test acc {},test loss {}".format(epoch, test_acc, test_loss))
            # writer.add_scalar('test/loss', test_loss, epoch)
            # writer.add_scalar('test/acc', test_acc, epoch)

    logger.info('Best val_acc: {} at {}'.format(best_acc, best_val_acc_loc))
    logger.info('Test acc: {}'.format(test_accs))


def validate(valloader, model, criterion, epoch, mode, cls_token_ids=None):
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
            # outputs = model(inputs, length)
            if cls_token_ids is None:
                outputs = model(inputs, length, probing_words_list=cls_token_ids, use_cls_sep=args.add_cls_sep)
            else:
                outputs, _ = model(inputs, length, probing_words_list=cls_token_ids, use_cls_sep=args.add_cls_sep)
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


def train(labeled_trainloader, labeled_train_iter, unlabeled_trainloader, unlabeled_train_iter, model, optimizer,
          criterion, epoch, n_labels, train_aug, cls_token_ids):
    model.train()
    if labeled_train_iter is None:
        labeled_train_iter = iter(labeled_trainloader)
    if unlabeled_train_iter is None:
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
        inputs_x, targets_x, inputs_x_length = inputs_x.cuda(), targets_x.cuda(
            non_blocking=True), inputs_x_length.cuda()
        assert args.split_match is True
        outputs_x_list, prob_logits_x = model(inputs_x, inputs_x_length, probing_words_list=cls_token_ids,
                                              use_cls_sep=args.add_cls_sep, split_match=args.split_match)
        # uda loss
        inputs_u, inputs_u2, inputs_ori = inputs_u.cuda(), inputs_u2.cuda(), inputs_ori.cuda()
        length_u, length_u2, length_ori = length_u.cuda(), length_u2.cuda(), length_ori.cuda()
        targets_u_ori = targets_u_ori.cuda()
        with torch.no_grad():
            orig_y_pred_list, prob_logits_orig = model(inputs_ori, length_ori, probing_words_list=cls_token_ids,
                                                       use_cls_sep=args.add_cls_sep, split_match=args.split_match)
            targets_u = 0
            for orig_y_pred in orig_y_pred_list:
                orig_y_pred = orig_y_pred / args.T
                targets_u += torch.softmax(orig_y_pred, dim=1)
            pl_u, pl_u_idx = torch.max(targets_u, 1)
            ipdb.set_trace()
            _, pl_u_idx_match = torch.max(prob_logits_orig, 1)
            loss_u_mask = (pl_u >= args.confid).float()
            loss_u_mask *= (pl_u_idx == pl_u_idx_match).float()
            writer.add_scalar('monitor/mask', loss_u_mask.mean(), batch_idx + args.val_iteration * epoch)
            pl_pred = targets_u.argmax(1)
            pl_acc = (pl_pred == targets_u_ori).float()
            writer.add_scalar('monitor/pl_acc', pl_acc.mean(), batch_idx + args.val_iteration * epoch)
            if loss_u_mask.sum() > 0:
                pl_acc = (pl_acc * loss_u_mask).sum() / loss_u_mask.sum()
            else:
                pl_acc = 0
            writer.add_scalar('monitor/mask_acc', pl_acc, batch_idx + args.val_iteration * epoch)
            prob_acc = (targets_u_ori == pl_u_idx_match).float().mean()
            writer.add_scalar('monitor/prob_acc', prob_acc, batch_idx + args.val_iteration * epoch)

        aug_y_pred_list, prob_logits_aug = model(inputs_u, length_u, probing_words_list=cls_token_ids,
                                                 use_cls_sep=args.add_cls_sep, split_match=args.split_match)
        aug_y_pred2_list, prob_logits_aug2 = model(inputs_u2, length_u2, probing_words_list=cls_token_ids,
                                                   use_cls_sep=args.add_cls_sep, split_match=args.split_match)

        loss = criterion(outputs_x_list, targets_x, prob_logits_x, (aug_y_pred_list, aug_y_pred2_list), orig_y_pred,
                         (prob_logits_aug, prob_logits_aug2), loss_u_mask, epoch + batch_idx / args.val_iteration)

        # ipdb.set_trace()
        writer.add_scalar('train/loss_total', loss[0], batch_idx + args.val_iteration * epoch)
        writer.add_scalar('train/loss_l', loss[1], batch_idx + args.val_iteration * epoch)
        writer.add_scalar('train/loss_uda', loss[2], batch_idx + args.val_iteration * epoch)
        writer.add_scalar('train/loss_l_prob', loss[3], batch_idx + args.val_iteration * epoch)
        writer.add_scalar('train/loss_u_prob', loss[4], batch_idx + args.val_iteration * epoch)
        if batch_idx % 100 == 0:
            logger.info(
                'epoch {}-{}, loss: total {}, l {}, ul {}, lprob {}, uprob {}, mask {} - {}'.format(epoch, batch_idx,
                                                                                                    loss[0], loss[1],
                                                                                                    loss[2], loss[3],
                                                                                                    loss[4],
                                                                                                    loss_u_mask.mean(),
                                                                                                    pl_acc))

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

    def __call__(self, outputs_x_list, targets_x, prob_logits_x, outputs_u=None, targets_u=None, prob_logits_u=None,
                 loss_u_mask=None, epoch=None):
        loss = 0
        for outputs_x in outputs_x_list:
            loss += self.ce_loss(outputs_x.view(-1, self.n_labels), targets_x.view(-1)).mean()
        loss /= len(outputs_x_list)
        assert args.tsa is False
        # if args.tsa:
        #     thres = self.TSA(epoch, self.n_labels, self.tsa_type)
        #     q_y_softmax = F.softmax(outputs_x, dim=1)
        #     targets_x_1hot = torch.zeros(outputs_x.shape[0], self.n_labels).scatter_(1, targets_x.cpu().view(-1, 1),
        #                                                                              1).cuda()
        #     q_correct = torch.sum(targets_x_1hot * q_y_softmax, dim=-1)
        #     loss_l_mask = (q_correct <= thres).float().detach()  # Ignore confident predictions.
        #     loss = (loss * loss_l_mask).sum() / max(1, loss_l_mask.sum())
        # else:
        #     loss = loss.mean()

        # loss_prob_x = self.bce_loss(self.sigmoid(prob_logits_x.view(-1)), targets_x_1hot.view(-1)).mean()
        loss_prob_x = self.ce_loss(prob_logits_x.view(-1, self.n_labels), targets_x.view(-1)).mean()

        # ul loss
        if loss_u_mask.sum() > 0:
            outputs_u1_list, outputs_u2_list = outputs_u
            loss_u1, loss_u2 = 0, 0
            for outputs_u1, outputs_u2 in zip(outputs_u1_list, outputs_u2_list):
                loss_u1 += (self.kl_divergence_from_logits(targets_u, outputs_u1) * loss_u_mask).sum() / loss_u_mask.sum()
                loss_u2 += (self.kl_divergence_from_logits(targets_u, outputs_u2) * loss_u_mask).sum() / loss_u_mask.sum()
            loss_uda = (loss_u1 + loss_u2) / 2 / len(outputs_u1_list) * args.lambda_u  # * self.linear_rampup(epoch)

            # ipdb.set_trace()

            prob_logits_u1, prob_logits_u2 = prob_logits_u
            targets_u_pl = torch.argmax(targets_u, 1)
            loss_prob_u1 = self.ce_loss(prob_logits_u1.view(-1, self.n_labels), targets_u_pl.view(-1))
            loss_prob_u1 = (loss_prob_u1 * loss_u_mask).sum() / loss_u_mask.sum()
            loss_prob_u2 = self.ce_loss(prob_logits_u2.view(-1, self.n_labels), targets_u_pl.view(-1))
            loss_prob_u2 = (loss_prob_u2 * loss_u_mask).sum() / loss_u_mask.sum()

            # bs_u, num_cls = outputs_u1.shape
            # loss_prob_mask = loss_u_mask.unsqueeze(1).expand(bs_u, num_cls).reshape(-1)
            # loss_prob_u1 = self.bce_loss(self.sigmoid(prob_logits_u1.view(-1)), torch.softmax(targets_u, dim=1).view(-1))
            # loss_prob_u1 = (loss_prob_u1 * loss_prob_mask).sum() / loss_prob_mask.sum()
            # loss_prob_u2 = self.bce_loss(self.sigmoid(prob_logits_u2.view(-1)), torch.softmax(targets_u, dim=1).view(-1))
            # loss_prob_u2 = (loss_prob_u2 * loss_prob_mask).sum() / loss_prob_mask.sum()
            loss_prob = (loss_prob_u1 + loss_prob_u2) / 2 * args.lambda_u
        else:
            loss_uda, loss_prob = 0, 0

        total_loss = loss + loss_uda + loss_prob_x + loss_prob
        return total_loss, loss, loss_uda, loss_prob_x, loss_prob


if __name__ == '__main__':
    main()
