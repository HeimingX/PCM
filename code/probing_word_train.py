import argparse
import os
import random
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from pytorch_transformers import *
from torch.autograd import Variable
from torch.utils.data import Dataset

from read_data import *
from normal_bert import ClassificationBert

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

parser.add_argument('--add_prob', action="store_true", default=False, help='adding probing words')
parser.add_argument('--add-cls-sep', action="store_true", default=False)
parser.add_argument('--classify-with-cls', action="store_true", default=False)

parser.add_argument('--resume', default="None", type=str, help='resume dir')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("gpu num: ", n_gpu)

best_acc = 0


def main():
    global best_acc
    train_labeled_set, train_unlabeled_set, val_set, test_set, n_labels, _ = get_data(
        args.data_path, args.n_labeled, add_prob=args.add_prob, add_cls_sep=args.add_cls_sep, mixText_origin=False)
    labeled_trainloader = Data.DataLoader(
        dataset=train_labeled_set, batch_size=args.batch_size, shuffle=True)
    val_loader = Data.DataLoader(
        dataset=val_set, batch_size=512, shuffle=False)
    test_loader = Data.DataLoader(
        dataset=test_set, batch_size=512, shuffle=False)


    model = ClassificationBert(n_labels).cuda()
    model = nn.DataParallel(model)
    optimizer = AdamW(
        [
            {"params": model.module.bert.parameters(), "lr": args.lrmain},
            {"params": model.module.linear.parameters(), "lr": args.lrlast},
        ])


    criterion = nn.CrossEntropyLoss()

    test_accs = []

    for epoch in range(args.epochs):
        train(labeled_trainloader, model, optimizer, criterion, epoch)

        # save model
        ckpt = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        filename = 'checkpoint.baseline_prob_word_with_cls_sep_ep{}.ckpt'.format(epoch)
        checkpoint_path = os.path.join('./ckpt/baseline_prob_word/', filename)
        torch.save(ckpt, checkpoint_path)

        # if args.resume and args.resume != "None":
        #     resume_filename = '{}_ep{}.ckpt'.format(args.resume, epoch)
        #     assert os.path.isfile(resume_filename), "=> no checkpoint found at '{}'".format(resume_filename)
        #     print("=> loading checkpoint '{}'".format(resume_filename))
        #     checkpoint = torch.load(resume_filename)
        #     model.load_state_dict(checkpoint['state_dict'])

        val_loss, val_acc = validate(
            val_loader, model, criterion, epoch, mode='Valid Stats')
        print("epoch {}, val acc {}, val_loss {}".format(
            epoch, val_acc, val_loss))

        if val_acc >= best_acc:
            best_acc = val_acc
            test_loss, test_acc = validate(
                test_loader, model, criterion, epoch, mode='Test Stats ')
            test_accs.append(test_acc)
            print("epoch {}, test acc {},test loss {}".format(
                epoch, test_acc, test_loss))

    print('Best val_acc:')
    print(best_acc)

    print('Test acc:')
    print(test_accs)


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

        for batch_idx, (inputs, targets, length) in enumerate(valloader):
            data_time += time.time() - end
            end = time.time()
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            _, predicted = torch.max(outputs.data, 1)

            correct += (np.array(predicted.cpu()) ==
                        np.array(targets.cpu())).sum()
            loss_total += loss.item() * inputs.shape[0]
            total_sample += inputs.shape[0]

            train_time += time.time() - end
            end = time.time()
        print('test time: data {}, infer {}'.format(data_time, train_time))

        acc_total = correct/total_sample
        loss_total = loss_total/total_sample

    return loss_total, acc_total


def train(labeled_trainloader, model, optimizer, criterion, epoch):
    model.train()

    data_time = 0
    train_time = 0
    end = time.time()
    for batch_idx, (inputs, targets, length, input_mask, segment_ids) in enumerate(labeled_trainloader):
        data_time += time.time() - end
        end = time.time()

        inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        input_mask, segment_ids = input_mask.cuda(), segment_ids.cuda()
        outputs = model(inputs, attention_mask=input_mask, token_type_ids=segment_ids, use_gap=args.classify_with_cls)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        print('epoch {}, step {}, loss {}'.format(
            epoch, batch_idx, loss.item()))
        loss.backward()
        optimizer.step()
        train_time += time.time() - end
        end = time.time()
    print('train time: data {}, infer {}'.format(data_time, train_time))


if __name__ == '__main__':
    main()