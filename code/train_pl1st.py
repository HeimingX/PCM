import argparse
import os
import random
from random import shuffle
import math
import time
import ipdb
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

import utils, retrieval_attention, global_file

parser = argparse.ArgumentParser(description='PyTorch MixText')

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
                    help='number of labeled data')

parser.add_argument('--un-labeled', default=5000, type=int,
                    help='number of unlabeled data')

parser.add_argument('--val-iteration', type=int, default=200,
                    help='number of labeled data')

parser.add_argument('--add-cls-sep', action="store_true", default=False)
parser.add_argument('--mix-option', default=True, type=bool, metavar='N',
                    help='mix option, whether to mix or not')
parser.add_argument('--mix-method', default=0, type=int, metavar='N',
                    help='mix method, set different mix method')
parser.add_argument('--separate-mix', default=False, type=bool, metavar='N',
                    help='mix separate from labeled data and unlabeled data')
parser.add_argument('--co', default=False, type=bool, metavar='N',
                    help='set a random choice between mix and unmix during training')
parser.add_argument('--train_aug', default=False, type=bool, metavar='N',
                    help='augment labeled training data')

parser.add_argument('--model', type=str, default='bert-base-uncased',
                    help='pretrained model')

parser.add_argument('--data-path', type=str, default='yahoo_answers_csv/',
                    help='path to data folders')

parser.add_argument('--mix-layers-set', nargs='+',
                    default=[0, 1, 2, 3], type=int, help='define mix layer set')

parser.add_argument('--alpha', default=0.75, type=float,
                    help='alpha for beta distribution')

parser.add_argument('--lambda-u', default=1, type=float,
                    help='weight for consistency loss term of unlabeled data')
parser.add_argument('--T', default=0.5, type=float,
                    help='temperature for sharpen function')

parser.add_argument('--temp-change', default=1000000, type=int)

parser.add_argument('--margin', default=0.7, type=float, metavar='N',
                    help='margin for hinge loss')
parser.add_argument('--lambda-u-hinge', default=0, type=float,
                    help='weight for hinge loss term of unlabeled data')

parser.add_argument('--resume', default="None", type=str, help='resume dir')
parser.add_argument('--specific_name', default=None, type=str, help='the specific name for the output file')
parser.add_argument('--eval', action="store_true", default=False)
parser.add_argument('--seed', default=0, type=float)
parser.add_argument('--seed_l', default=0, type=int)
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
from mixtext import MixText

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("gpu num: %s", n_gpu)

best_acc = 0
total_steps = 0
flag = 0
logger.info('Whether mix: {}'.format(args.mix_option))
logger.info("Mix layers sets: {}".format(args.mix_layers_set))


def main():
    global best_acc
    # Read dataset and build dataloaders
    train_labeled_set, train_unlabeled_set, val_set, test_set, n_labels, _ = get_data(
        args.data_path, args.n_labeled, args.un_labeled, model=args.model, train_aug=args.train_aug,
        add_cls_sep=args.add_cls_sep, mixText_origin=True, splited=False)
    args.n_labels = n_labels
    labeled_trainloader = Data.DataLoader(dataset=train_labeled_set, batch_size=args.batch_size, shuffle=True)
    unlabeled_trainloader = Data.DataLoader(dataset=train_unlabeled_set, batch_size=args.batch_size_u, shuffle=True)
    val_loader = Data.DataLoader(dataset=val_set, batch_size=128, shuffle=False)
    test_loader = Data.DataLoader(dataset=test_set, batch_size=128, shuffle=False)

    # Define the model, set the optimizer
    model = MixText(n_labels, args.mix_option, require_attention=True).cuda()
    model = nn.DataParallel(model)
    optimizer = AdamW(
        [
            {"params": model.module.bert.parameters(), "lr": args.lrmain},
            {"params": model.module.linear.parameters(), "lr": args.lrlast},
        ])

    num_warmup_steps = math.floor(50)
    num_total_steps = args.val_iteration

    scheduler = None
    # WarmupConstantSchedule(optimizer, warmup_steps=num_warmup_steps)

    train_criterion = SemiLoss()
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
        writer.add_scalar('test/loss', test_loss, epoch)
        writer.add_scalar('test/acc', test_acc, epoch)
        ipdb.set_trace()
        return
    elif args.resume and args.resume != "None":
        checkpoint_path = args.resume
        assert os.path.isfile(checkpoint_path), "=> no checkpoint found at '{}'".format(checkpoint_path)
        logger.info("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    test_accs = []
    best_val_acc_loc = 0
    ce_criterion = nn.CrossEntropyLoss()
    args.val_iteration = len(unlabeled_trainloader)
    # Start training
    for epoch in range(args.epochs):
        ul_pl, ul_pl_prob = update_pl(train_unlabeled_set, model, epoch)
        ul_pl_stats = (ul_pl, ul_pl_prob)
        train(labeled_trainloader, unlabeled_trainloader, model, optimizer, scheduler, train_criterion, epoch, n_labels,
              args.train_aug, ul_pl_stats)

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

    logger.info("Finished training!")
    logger.info('Best val_acc: {} at {}'.format(best_acc, best_val_acc_loc))
    logger.info('Test acc: {}'.format(test_accs))


def update_pl(dataset, model, epoch):
    model.eval()
    num_ul = args.n_labels * args.un_labeled
    ul_gt = torch.ones(num_ul).long().cuda() * -1
    ul_pl = torch.ones(num_ul).long().cuda() * -1
    ul_pl_prob = torch.zeros((num_ul, args.n_labels)).cuda()
    pos_idx = list()
    neg_idx = list()
    dl = Data.DataLoader(dataset=dataset, batch_size=128, shuffle=False, drop_last=False)
    with torch.no_grad():
        for batch_idx, ((inputs_u, inputs_u2, inputs_ori), (_, _, _), idx_u, targets_u_ori) in enumerate(dl):
            inputs_ori = inputs_ori.cuda()
            targets_u_ori = targets_u_ori.cuda()
            ul_gt[idx_u] = targets_u_ori

            # Predict labels for unlabeled data.
            # Based on translation qualities, choose different weights here.
            # For AG News: German: 1, Russian: 0, ori: 1
            # For DBPedia: German: 1, Russian: 1, ori: 1
            # For IMDB: German: 0, Russian: 0, ori: 1
            # For Yahoo Answers: German: 1, Russian: 0, ori: 1 / German: 0, Russian: 0, ori: 1
            outputs_ori = model(inputs_ori)
            if 'yahoo_answers' in args.data_path:
                p = torch.softmax(outputs_ori, dim=1)
            elif 'ag_news' in args.data_path:
                outputs_u = model(inputs_u)
                p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_ori, dim=1)) / 2
            elif 'dbpedia' in args.data_path:
                outputs_u = model(inputs_u)
                outputs_u2 = model(inputs_u2)
                p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1) + torch.softmax(outputs_ori,
                                                                                                        dim=1)) / 3
            elif 'imdb' in args.data_path:
                p = torch.softmax(outputs_ori, dim=1)
            else:
                raise LookupError
            # Do a sharpen here.
            pt = p ** (1 / args.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()
            ul_pl_prob[idx_u] = targets_u

            _, pl_u_idx = torch.max(targets_u, 1)
            ul_pl[idx_u] = pl_u_idx

            # get pos/neg pl idx
            pos_mask = (pl_u_idx == 1)
            pos_idx.extend(idx_u[pos_mask].tolist())
            neg_mask = pl_u_idx == 0
            neg_idx.extend(idx_u[neg_mask].tolist())

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
    return ul_pl, ul_pl_prob


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, scheduler, criterion, epoch, n_labels,
          train_aug=False, ul_pl_stats=None):
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    ul_pl, ul_pl_prob = ul_pl_stats
    model.train()

    global total_steps
    global flag
    if flag == 0 and total_steps > args.temp_change:
        logger.info('Change T!')
        args.T = 0.9
        flag = 1
    data_time = 0
    train_time = 0
    end = time.time()
    for batch_idx in range(args.val_iteration):

        total_steps += 1

        if not train_aug:
            try:
                inputs_x, targets_x, inputs_x_length, _ = labeled_train_iter.next()
            except:
                labeled_train_iter = iter(labeled_trainloader)
                inputs_x, targets_x, inputs_x_length, _ = labeled_train_iter.next()
        else:
            try:
                (inputs_x, inputs_x_aug), (targets_x, _), (inputs_x_length,
                                                           inputs_x_length_aug) = labeled_train_iter.next()
            except:
                labeled_train_iter = iter(labeled_trainloader)
                (inputs_x, inputs_x_aug), (targets_x, _), (inputs_x_length,
                                                           inputs_x_length_aug) = labeled_train_iter.next()
        try:
            (inputs_u, inputs_u2, inputs_ori), (length_u, length_u2, length_ori), idx_u, targets_u_ori = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2, inputs_ori), (length_u, length_u2, length_ori), idx_u, targets_u_ori = unlabeled_train_iter.next()
        data_time += time.time() - end
        end = time.time()

        batch_size = inputs_x.size(0)
        batch_size_2 = inputs_ori.size(0)
        targets_x = torch.zeros(batch_size, n_labels).scatter_(1, targets_x.view(-1, 1), 1)

        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
        inputs_u = inputs_u.cuda()
        inputs_u2 = inputs_u2.cuda()
        inputs_ori = inputs_ori.cuda()

        targets_u = ul_pl_prob[idx_u]
        # mask = []
        # with torch.no_grad():
        #     # Predict labels for unlabeled data.
        #     # Based on translation qualities, choose different weights here.
        #     # For AG News: German: 1, Russian: 0, ori: 1
        #     # For DBPedia: German: 1, Russian: 1, ori: 1
        #     # For IMDB: German: 0, Russian: 0, ori: 1
        #     # For Yahoo Answers: German: 1, Russian: 0, ori: 1 / German: 0, Russian: 0, ori: 1
        #     outputs_ori = model(inputs_ori)
        #     if 'yahoo_answers' in args.data_path:
        #         p = torch.softmax(outputs_ori, dim=1)
        #     elif 'ag_news' in args.data_path:
        #         outputs_u = model(inputs_u)
        #         p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_ori, dim=1)) / 2
        #     elif 'dbpedia' in args.data_path:
        #         outputs_u = model(inputs_u)
        #         outputs_u2 = model(inputs_u2)
        #         p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1) + torch.softmax(outputs_ori,
        #                                                                                                 dim=1)) / 3
        #     elif 'imdb' in args.data_path:
        #         p = torch.softmax(outputs_ori, dim=1)
        #     else:
        #         raise LookupError
        #     # Do a sharpen here.
        #     pt = p ** (1 / args.T)
        #     targets_u = pt / pt.sum(dim=1, keepdim=True)
        #     targets_u = targets_u.detach()

        mixed = 1

        if args.co:
            mix_ = np.random.choice([0, 1], 1)[0]
        else:
            mix_ = 1

        if mix_ == 1:
            l = np.random.beta(args.alpha, args.alpha)
            if args.separate_mix:
                l = l
            else:
                l = max(l, 1 - l)
        else:
            l = 1

        mix_layer = np.random.choice(args.mix_layers_set, 1)[0]
        mix_layer = mix_layer - 1

        if not train_aug:
            if mix_ == 1:
                satisfy_mask = ul_pl[idx_u] > -1
                _inputs_u = inputs_u[satisfy_mask]
                _inputs_u2 = inputs_u2[satisfy_mask]
                _inputs_ori = inputs_ori[satisfy_mask]
                _length_u = length_u[satisfy_mask]
                _length_u2 = length_u2[satisfy_mask]
                _length_ori = length_ori[satisfy_mask]
                _targets_u = targets_u[satisfy_mask]
                all_inputs = torch.cat([inputs_x, _inputs_u, _inputs_u2, _inputs_ori], dim=0)
                all_lengths = torch.cat([inputs_x_length, _length_u, _length_u2, _length_ori], dim=0)
                all_targets = torch.cat([targets_x, _targets_u, _targets_u, _targets_u], dim=0)
            else:
                all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2, inputs_ori, inputs_ori], dim=0)
                all_lengths = torch.cat([inputs_x_length, length_u, length_u2, length_ori, length_ori], dim=0)
                all_targets = torch.cat([targets_x, targets_u, targets_u, targets_u, targets_u], dim=0)
        else:
            all_inputs = torch.cat(
                [inputs_x, inputs_x_aug, inputs_u, inputs_u2, inputs_ori], dim=0)
            all_lengths = torch.cat(
                [inputs_x_length, inputs_x_length, length_u, length_u2, length_ori], dim=0)
            all_targets = torch.cat(
                [targets_x, targets_x, targets_u, targets_u, targets_u], dim=0)

        if args.separate_mix:
            idx1 = torch.randperm(batch_size)
            idx2 = torch.randperm(all_inputs.size(0) - batch_size) + batch_size
            idx = torch.cat([idx1, idx2], dim=0)
        else:
            if mix_ == 1:
                idx = torch.randperm(all_inputs.size(0))
            else:
                idx1 = torch.randperm(all_inputs.size(0) - batch_size_2)
                idx2 = torch.arange(batch_size_2) + all_inputs.size(0) - batch_size_2
                idx = torch.cat([idx1, idx2], dim=0)

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        length_a, length_b = all_lengths, all_lengths[idx]

        if args.mix_method == 0:
            # Mix sentences' hidden representations
            # input_a_sp = torch.split(input_a, split_size=9, dim=0)
            # input_b_sp = torch.split(input_b, split_size=9, dim=0)
            # logits = list()
            # for _, (_input_a, _input_b) in enumerate(input_a_sp, input_b_sp):
            #     logits.append(model(_input_a, _input_b, l, mix_layer))
            # logits = torch.cat(logits, 0)
            # mixed_target = l * target_a + (1 - l) * target_b

            logits = model(input_a, input_b, l, mix_layer)
            mixed_target = l * target_a + (1 - l) * target_b

        elif args.mix_method == 1:
            # Concat snippet of two training sentences, the snippets are selected based on l
            # For example: "I lova you so much" and "He likes NLP" could be mixed as "He likes NLP so much".
            # The corresponding labels are mixed with coefficient as well
            mixed_input = []
            if l != 1:
                for i in range(input_a.size(0)):
                    length1 = math.floor(int(length_a[i]) * l)
                    idx1 = torch.randperm(int(length_a[i]) - length1 + 1)[0]
                    length2 = math.ceil(int(length_b[i]) * (1 - l))
                    if length1 + length2 > 256:
                        length2 = 256 - length1 - 1
                    idx2 = torch.randperm(int(length_b[i]) - length2 + 1)[0]
                    try:
                        mixed_input.append(
                            torch.cat((input_a[i][idx1: idx1 + length1], torch.tensor([102]).cuda(),
                                       input_b[i][idx2:idx2 + length2],
                                       torch.tensor([0] * (256 - 1 - length1 - length2)).cuda()), dim=0).unsqueeze(0))
                    except:
                        print(256 - 1 - length1 - length2,
                              idx2, length2, idx1, length1)

                mixed_input = torch.cat(mixed_input, dim=0)

            else:
                mixed_input = input_a

            logits = model(mixed_input)
            mixed_target = l * target_a + (1 - l) * target_b

        elif args.mix_method == 2:
            # Concat two training sentences
            # The corresponding labels are averaged
            if l == 1:
                mixed_input = []
                for i in range(input_a.size(0)):
                    mixed_input.append(
                        torch.cat((input_a[i][:length_a[i]], torch.tensor([102]).cuda(), input_b[i][:length_b[i]],
                                   torch.tensor([0] * (512 - 1 - int(length_a[i]) - int(length_b[i]))).cuda()),
                                  dim=0).unsqueeze(0))

                mixed_input = torch.cat(mixed_input, dim=0)
                logits = model(mixed_input, sent_size=512)

                # mixed_target = torch.clamp(target_a + target_b, max = 1)
                mixed = 0
                mixed_target = (target_a + target_b) / 2
            else:
                mixed_input = input_a
                mixed_target = target_a
                logits = model(mixed_input, sent_size=256)
                mixed = 1

        # Lx, Lu, w, Lu2, w2 = criterion(logits[:batch_size], mixed_target[:batch_size], logits[batch_size:-batch_size_2],
        #                                mixed_target[batch_size:-batch_size_2], logits[-batch_size_2:], epoch+batch_idx/args.val_iteration, mixed)
        if mix_ == 1:

            Lx, Lu, w, Lu2, w2 = criterion(logits[:batch_size], mixed_target[:batch_size],
                                           logits[batch_size:], mixed_target[batch_size:], None,
                                           epoch + batch_idx / args.val_iteration, mixed)
            loss = Lx + w * Lu
        else:
            Lx, Lu, w, Lu2, w2 = criterion(logits[:batch_size], mixed_target[:batch_size],
                                           logits[batch_size:-batch_size_2],
                                           mixed_target[batch_size:-batch_size_2], logits[-batch_size_2:],
                                           epoch + batch_idx / args.val_iteration, mixed)
            loss = Lx + w * Lu + w2 * Lu2

        # max_grad_norm = 1.0
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        if batch_idx % 100 == 0:
            # logger.info("epoch {}, step {}, loss {}, Lx {}, Lu {}, Lu2 {}".format(
            #     epoch, batch_idx, loss.item(), Lx.item(), Lu.item(), Lu2.item()))
            logger.info("epoch {}, step {}, loss {}, Lx {}, Lu {}".format(epoch, batch_idx, loss.item(), Lx.item(), Lu.item()))
        train_time += time.time() - end
        end = time.time()
    logger.info('---> train time: data {}, infer {}'.format(data_time, train_time))


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

        # for batch_idx, (inputs, targets, length, _, _, _) in enumerate(valloader):
        for batch_idx, (inputs, targets, length, _) in enumerate(valloader):
            data_time += time.time() - end
            end = time.time()
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            _, predicted = torch.max(outputs.data, 1)

            if batch_idx == 0:
                logger.info("Sample some true labeles and predicted labels")
                logger.info(predicted[:20])
                logger.info(targets[:20])

            correct += (np.array(predicted.cpu()) == np.array(targets.cpu())).sum()
            loss_total += loss.item() * inputs.shape[0]
            total_sample += inputs.shape[0]

            train_time += time.time() - end
            end = time.time()
        logger.info('---> test time: data {}, infer {}'.format(data_time, train_time))

        acc_total = correct / total_sample
        loss_total = loss_total / total_sample

    return loss_total, acc_total


def get_gradcam(valloader, model, criterion, epoch, mode):
    model.eval()
    data_time = 0
    train_time = 0
    end = time.time()

    loss_total = 0
    total_sample = 0
    acc_total = 0
    correct = 0

    false_id_set = list()
    false_prediction_set = list()
    prediction_set = list()
    target_set = list()
    cam_set = list()
    attention_set = list()

    for batch_idx, (inputs, targets, length, input_mask, segment_ids, idx) in enumerate(valloader):
        print(batch_idx)
        data_time += time.time() - end
        end = time.time()
        inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        outputs = model(inputs, register_hook=True)
        loss = criterion(outputs, targets)
        # ipdb.set_trace()

        _, predicted = torch.max(outputs.data, 1)

        if batch_idx == 0:
            logger.info("Sample some true labeles and predicted labels")
            logger.info(predicted[:20])
            logger.info(targets[:20])

        correct += (np.array(predicted.cpu()) == np.array(targets.cpu())).sum()
        loss_total += loss.item() * inputs.shape[0]
        total_sample += inputs.shape[0]

        correct_prediction = np.array(predicted.cpu()) == np.array(targets.cpu())
        false_id_set.append(idx[~correct_prediction])
        pred_prob = torch.softmax(outputs.data, 1)
        false_prediction_set.append(pred_prob[~correct_prediction].cpu())
        prediction_set.append(pred_prob.cpu())
        target_set.append(targets.cpu())

        output_value, _ = torch.max(outputs, 1)
        all_values = output_value.sum()
        all_values.backward()
        # gradients = model.module.get_activations_gradient().detach()
        gradients = model.get_activations_gradient().detach().cpu()
        pooled_gradients = torch.mean(gradients, dim=[0, 1], keepdim=True)
        # activations = model.module.get_activations().detach()
        activations = model.get_activations().detach().cpu()
        bs, seq_len, chn = activations.shape
        weighted_act = pooled_gradients.repeat(bs, seq_len, 1) * activations
        heatmap = torch.mean(weighted_act, dim=2).squeeze()
        cam_set.append(heatmap)

        # attention_set.append(model.module.get_attentions())
        attention_set.append(model.get_attentions()[-1].detach().cpu())
        model.zero_grad()
        # import ipdb; ipdb.set_trace()

        train_time += time.time() - end
        end = time.time()
    logger.info('---> test time: data {}, infer {}'.format(data_time, train_time))

    acc_total = correct / total_sample
    loss_total = loss_total / total_sample

    detail_res = {
        'f_p': torch.cat(false_prediction_set, 0),
        'f_id': torch.cat(false_id_set),
        'all_p': torch.cat(prediction_set, 0),
        'all_l': torch.cat(target_set),
        'all_cam': torch.cat(cam_set, 0),
        'all_attention': attention_set
    }

    return loss_total, acc_total, detail_res


def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, outputs_u_2, epoch, mixed=1):

        if args.mix_method == 0 or args.mix_method == 1:
            Lx = - torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
            probs_u = torch.softmax(outputs_u, dim=1)
            Lu = F.kl_div(probs_u.log(), targets_u, None, None, 'batchmean')
            # Lu2 = torch.mean(torch.clamp(torch.sum(-F.softmax(outputs_u, dim=1)
            #                                        * F.log_softmax(outputs_u, dim=1), dim=1) - args.margin, min=0))
            if outputs_u_2 is None:
                Lu2 = 0
            else:
                Lu2 = torch.mean(torch.clamp(torch.sum(-F.softmax(outputs_u_2, dim=1)
                                                       * F.log_softmax(outputs_u_2, dim=1), dim=1) - args.margin,
                                             min=0))
        elif args.mix_method == 2:
            if mixed == 0:
                Lx = - \
                    torch.mean(torch.sum(F.logsigmoid(
                        outputs_x) * targets_x, dim=1))

                probs_u = torch.softmax(outputs_u, dim=1)

                Lu = F.kl_div(probs_u.log(), targets_u,
                              None, None, 'batchmean')

                Lu2 = torch.mean(torch.clamp(args.margin - torch.sum(
                    F.softmax(outputs_u_2, dim=1) * F.softmax(outputs_u_2, dim=1), dim=1), min=0))
            else:
                Lx = - \
                    torch.mean(torch.sum(F.log_softmax(
                        outputs_x, dim=1) * targets_x, dim=1))

                probs_u = torch.softmax(outputs_u, dim=1)
                Lu = F.kl_div(probs_u.log(), targets_u,
                              None, None, 'batchmean')

                Lu2 = torch.mean(torch.clamp(args.margin - torch.sum(
                    F.softmax(outputs_u, dim=1) * F.softmax(outputs_u, dim=1), dim=1), min=0))

        return Lx, Lu, args.lambda_u * linear_rampup(epoch), Lu2, args.lambda_u_hinge * linear_rampup(epoch)


if __name__ == '__main__':
    main()
