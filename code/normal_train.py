import argparse
import os
import random
import math
import time
import logging
import shutil
import ipdb

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

parser = argparse.ArgumentParser(description='PyTorch Base Models')

parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=4, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--batch-size-u', default=24, type=int, metavar='N',
                    help='train batchsize')

parser.add_argument('--lrmain', '--learning-rate-bert', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate for bert')
parser.add_argument('--lrlast', '--learning-rate-model', default=0.001, type=float,
                    metavar='LR', help='initial learning rate for models')

parser.add_argument('--gpu', default='0,1,2,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--n-labeled', type=int, default=20,
                    help='Number of labeled data')
parser.add_argument('--val-iteration', type=int, default=200,
                    help='Number of labeled data')

parser.add_argument('--mix-option', default=False, type=bool, metavar='N',
                    help='mix option')
parser.add_argument('--train_aug', default=False, type=bool, metavar='N',
                    help='aug for training data')

parser.add_argument('--model', type=str, default='bert-base-uncased',
                    help='pretrained model')

parser.add_argument('--data-path', type=str, default='yahoo_answers_csv/',
                    help='path to data folders')

parser.add_argument('--add-cls-sep', action="store_true", default=False)
parser.add_argument('--classify-with-cls', action="store_true", default=False)
parser.add_argument('--gap-with-cls', action="store_true", default=False)

parser.add_argument('--prob_topk', default=30, type=float, help="selecting percent of probing words from sentence")

parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--seed_l', default=0, type=int)
parser.add_argument('--resume', default="None", type=str, help='resume dir')
parser.add_argument('--specific_name', default=None, type=str, help='the specific name for the output file')
parser.add_argument('--prob_save_sp', default='', type=str, help='the specific name for probing words saving')
parser.add_argument('--eval', action="store_true", default=False)
args = parser.parse_args()

global_file.init_args()
global_file.init_probing_words_len_list()
global_file.args = args

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


def main():
    global best_acc
    train_labeled_set, train_unlabeled_set, val_set, test_set, n_labels, _ = get_data(
        args.data_path, args.n_labeled, add_cls_sep=args.add_cls_sep, mixText_origin=False)
    labeled_trainloader = Data.DataLoader(dataset=train_labeled_set, batch_size=args.batch_size, shuffle=True)
    val_loader = Data.DataLoader(dataset=val_set, batch_size=128, shuffle=False)
    test_loader = Data.DataLoader(dataset=test_set, batch_size=128, shuffle=False)

    model = ClassificationBert(n_labels, use_gap=not args.classify_with_cls, use_cls_sep=args.gap_with_cls,
                               require_attention=True).cuda()

    model = nn.DataParallel(model)
    optimizer = AdamW(
        [
            {"params": model.module.bert.parameters(), "lr": args.lrmain},
            {"params": model.module.classifier.parameters(), "lr": args.lrlast},
        ])

    criterion = nn.CrossEntropyLoss()

    if args.eval:
        checkpoint_path = args.resume
        assert os.path.isfile(checkpoint_path), "=> no checkpoint found at '{}'".format(checkpoint_path)
        logger.info("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        epoch = checkpoint['epoch']

        val_loss, val_acc = validate(val_loader, model, criterion, epoch, mode='Valid Stats')
        logger.info("epoch {}, val acc {}, val_loss {}".format(epoch, val_acc, val_loss))
        test_loss, test_acc = validate(test_loader, model, criterion, epoch, mode='Test Stats ')
        logger.info("epoch {}, test acc {},test loss {}".format(epoch, test_acc, test_loss))

        # ====== get attended tokens ======
        l_dl = Data.DataLoader(dataset=train_labeled_set, batch_size=args.batch_size, shuffle=False, drop_last=False)
        l_attention, l_y_pred = retrieval_attention.get_attention_all(l_dl, model, data_num=args.n_labeled * n_labels,
                                                                      ul_flag=False, split_tf=False, nlabel=n_labels)
        texts_set = train_labeled_set.text
        tokens_set = train_labeled_set.train_tokens
        attention_set = l_attention
        # y_pred_set = l_y_pred
        y_pred_set = train_labeled_set.labels
        topk = args.prob_topk
        savename = args.data_path + 'ft'
        savename += '_nlab' + str(args.n_labeled)
        savename += '_top' + str(int(topk)) + '_ep' + str(epoch) + args.prob_save_sp + '.npy'
        attended_set = retrieval_attention.extract_class_wise_attened_tokens_ft(texts_set, tokens_set, attention_set,
                                                                                y_pred_set, n_labels, epoch, savename,
                                                                                topk)
        attened_set_len = [len(attended_set[i]) for i in range(len(attended_set))]
        print('attended_set len: {}'.format(attened_set_len))
        # ipdb.set_trace()
        return

    test_accs = []
    best_val_acc_loc = 0
    for epoch in range(args.epochs):
        train(labeled_trainloader, model, optimizer, criterion, epoch)
        # save model
        ckpt = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        filename = 'checkpoint.{}.ckpt'.format(run_context.exp_name)
        checkpoint_path = os.path.join(run_context.ckpt_dir, filename)
        torch.save(ckpt, checkpoint_path)

        val_loss, val_acc = validate(val_loader, model, criterion, epoch, mode='Valid Stats')
        logger.info("epoch {}, val acc {}, val_loss {}".format(epoch, val_acc, val_loss))
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)

        if val_acc >= best_acc:
            best_acc = val_acc
            best_val_acc_loc = epoch
            test_loss, test_acc = validate(test_loader, model, criterion, epoch, mode='Test Stats ')
            logger.info("epoch {}, test acc {},test loss {}".format(epoch, test_acc, test_loss))

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
        correct = 0

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

            train_time += time.time() - end
            end = time.time()
    logger.info('test time: data {}, infer {}'.format(data_time, train_time))

    acc_total = correct / total_sample
    loss_total = loss_total / total_sample

    return loss_total, acc_total


def train(labeled_trainloader, model, optimizer, criterion, epoch):
    model.train()

    iter_len = len(labeled_trainloader)
    l_iter = iter(labeled_trainloader)

    data_time = 0
    train_time = 0
    end = time.time()
    for batch_idx in range(iter_len):
        inputs, targets, length, _, _, idx = next(l_iter)
        data_time += time.time() - end
        end = time.time()

        inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        length = length.cuda()
        outputs = model(inputs, length=length)
        loss = criterion(outputs, targets)
        writer.add_scalar('train/loss_l', loss.item(), batch_idx + iter_len * epoch)

        optimizer.zero_grad()
        logger.info('epoch {}, step {}, loss {}'.format(epoch, batch_idx, loss.item()))
        loss.backward()
        optimizer.step()
        train_time += time.time() - end
        end = time.time()
    logger.info('train time: data {}, infer {}'.format(data_time, train_time))


if __name__ == '__main__':
    main()
