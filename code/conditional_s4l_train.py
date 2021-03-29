import argparse
import os
import random
import math
import time
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

import utils, retrieval_attention

parser = argparse.ArgumentParser(description='PyTorch MixText')

parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch-size', default=4, type=int, metavar='N', help='train batchsize')
parser.add_argument('--batch-size-u', default=24, type=int, metavar='N', help='train batchsize')

parser.add_argument('--lrmain', '--learning-rate-bert', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate for bert')
parser.add_argument('--lrlast', '--learning-rate-model', default=0.001, type=float,
                    metavar='LR', help='initial learning rate for models')

parser.add_argument('--gpu', default='0,1,2,3', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--n-labeled', type=int, default=20, help='number of labeled data')
parser.add_argument('--un-labeled', default=5000, type=int, help='number of unlabeled data')
parser.add_argument('--val-iteration', type=int, default=200, help='number of labeled data')

parser.add_argument('--mix-option', default=True, type=bool, metavar='N', help='mix option, whether to mix or not')
parser.add_argument('--mix-method', default=0, type=int, metavar='N', help='mix method, set different mix method')
parser.add_argument('--separate-mix', default=False, type=bool, metavar='N',
                    help='mix separate from labeled data and unlabeled data')
parser.add_argument('--co', default=False, type=bool, metavar='N',
                    help='set a random choice between mix and unmix during training')
parser.add_argument('--train_aug', default=False, type=bool, metavar='N', help='augment labeled training data')

parser.add_argument('--model', type=str, default='bert-base-uncased', help='pretrained model')
parser.add_argument('--data-path', type=str, default='yahoo_answers_csv/', help='path to data folders')
parser.add_argument('--mix-layers-set', nargs='+', default=[0, 1, 2, 3], type=int, help='define mix layer set')
parser.add_argument('--alpha', default=0.75, type=float, help='alpha for beta distribution')

parser.add_argument('--lambda-u', default=1, type=float, help='weight for consistency loss term of unlabeled data')
parser.add_argument('--T', default=0.5, type=float, help='temperature for sharpen function')
parser.add_argument('--temp-change', default=1000000, type=int)

parser.add_argument('--margin', default=0.7, type=float, metavar='N', help='margin for hinge loss')
parser.add_argument('--lambda-u-hinge', default=0, type=float,
                    help='weight for hinge loss term of unlabeled data')
# MLM
parser.add_argument("--short_seq_prob", type=float, default=0.1,
                    help="Probability of making a short sentence as a training example")
parser.add_argument("--masked_lm_prob", type=float, default=0.15,
                    help="Probability of masking each token for the LM task")
parser.add_argument("--max_predictions_per_seq", type=int, default=20,
                    help="Maximum number of tokens to mask in each sequence")
parser.add_argument("--do_whole_word_mask", action="store_true",
                    help="Whether to use whole word masking rather than per-WordPiece masking.")
parser.add_argument("--mlm_batch_size", default=32, type=int, help="Total batch size for training.")
parser.add_argument('--lrmlm', '--learning-rate-mlm', default=2e-5, type=float,
                    metavar='LR', help='initial learning rate for models')
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

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
from mixtext import MixText

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("gpu num: %s", n_gpu)

best_acc = 0
total_steps = 0
flag = 0
vocab_size = 30522
attention = torch.zeros(((args.n_labeled + args.un_labeled) * 10, 256))
# Based on translation qualities, choose different weights here.
# For AG News: German: 1, Russian: 0, ori: 1
# For DBPedia: German: 1, Russian: 1, ori: 1
# For IMDB: German: 0, Russian: 0, ori: 1
# For Yahoo Answers: German: 1, Russian: 0, ori: 1 / German: 0, Russian: 0, ori: 1
pl_gen_weight = {
    "yahoo": [0, 0, 1],
    "agnews": [1, 0, 1],
    'imdb': [0, 0, 1],
    'dbpedia': [1, 1, 1]
}
logger.info('Whether mix: {}'.format(args.mix_option))
logger.info("Mix layers sets: {}".format(args.mix_layers_set))


def main():
    global best_acc
    # Read dataset and build dataloaders
    train_labeled_set, train_unlabeled_set, val_set, test_set, mlm_set, n_labels, _ = get_data(
        args.data_path, args.n_labeled, args.un_labeled, model=args.model, train_aug=args.train_aug,
        mixText_origin=True, mlm_data=True, masked_lm_prob=args.masked_lm_prob,
        max_predictions_per_seq=args.max_predictions_per_seq, do_whole_word_mask=args.do_whole_word_mask)
    labeled_trainloader = Data.DataLoader(dataset=train_labeled_set, batch_size=args.batch_size, shuffle=True,
                                          drop_last=True)
    unlabeled_trainloader = Data.DataLoader(dataset=train_unlabeled_set, batch_size=args.batch_size_u, shuffle=True,
                                            drop_last=True)
    val_loader = Data.DataLoader(dataset=val_set, batch_size=256, shuffle=False)
    test_loader = Data.DataLoader(dataset=test_set, batch_size=256, shuffle=False)

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

    # WarmupConstantSchedule(optimizer, warmup_steps=num_warmup_steps)
    scheduler = None

    # MLM optim, refer: lm_finetuning/simple_lm_finetuning.py, line 542
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer_mlm = AdamW(optimizer_grouped_parameters, lr=args.lrmlm, eps=args.adam_epsilon)
    # refer: https://medium.com/@aniruddha.choudhury94/part-2-bert-fine-tuning-tutorial-with-pytorch-for-text-classification-on-the-corpus-of-linguistic-18057ce330e1
    # optimizer_mlm = AdamW([{"params": model.module.bert.parameters(), "lr": args.lrmlm}])
    scheduler_mlm = WarmupLinearSchedule(optimizer_mlm, warmup_steps=args.warmup_steps, t_total=int(args.val_iteration * args.epochs))

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()

    if args.eval:
        checkpoint_path = args.resume
        assert os.path.isfile(checkpoint_path), "=> no checkpoint found at '{}'".format(checkpoint_path)
        logger.info("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        epoch = checkpoint['epoch']
        # if isinstance(model, torch.nn.DataParallel):
        #     model = model.module
        ipdb.set_trace()
        test_loss, test_acc = validate(test_loader, model, criterion, epoch, mode='Test Stats ')
        # test_loss, test_acc, detail_res = get_gradcam(test_loader, model, criterion, epoch, mode='Test Stats ')
        logger.info("epoch {}, test acc {},test loss {}".format(epoch, test_acc, test_loss))

        # calculated attented tokens in each split set
        # l_dl = Data.DataLoader(dataset=train_labeled_set, batch_size=100, shuffle=False, drop_last=False)
        # l_attention, l_y_pred = retrieval_attention.get_attention_all(l_dl, model, data_num=args.n_labeled*n_labels)
        # file_name = 'conditional_s4l_l.csv'
        # retrieval_attention.construct_attention_file(train_labeled_set.text, train_labeled_set.train_tokens,
        #                                              l_attention, train_labeled_set.labels, l_y_pred, file_name,
        #                                              AGNEWS_LABEL_NAMES)

        # ul_dl = Data.DataLoader(dataset=train_unlabeled_set, batch_size=128, shuffle=False, drop_last=False)
        # u_attention, u_y_pred = retrieval_attention.get_attention_all(ul_dl, model, data_num=args.un_labeled * n_labels,
        #                                                               ul_flag=True)
        # file_name = 'conditional_s4l_ul.csv'
        # retrieval_attention.construct_attention_file(train_unlabeled_set.text, train_unlabeled_set.train_tokens,
        #                                              u_attention, train_unlabeled_set.labels, u_y_pred, file_name,
        #                                              AGNEWS_LABEL_NAMES)

        # test_dl = Data.DataLoader(dataset=test_set, batch_size=128, shuffle=False, drop_last=False)
        # test_attention, test_y_pred = retrieval_attention.get_attention_all(test_dl, model, data_num=60000)
        # file_name = 'conditional_s4l_test.csv'
        # retrieval_attention.construct_attention_file(test_set.text, test_set.train_tokens, test_attention,
        #                                              test_set.labels, test_y_pred, file_name, AGNEWS_LABEL_NAMES)
        # calculate attened tokens in each class
        # texts_set = (train_labeled_set.text, train_unlabeled_set.text)
        # tokens_set = (train_labeled_set.train_tokens, train_unlabeled_set.train_tokens)
        # attention_set = (l_attention, u_attention)
        # y_pred_set = (l_y_pred, u_y_pred)
        # cls_attended_tokens = retrieval_attention.extract_class_wise_attened_tokens(texts_set, tokens_set, attention_set, y_pred_set, AGNEWS_LABEL_NAMES)
        ipdb.set_trace()
        # train_tokens = test_set.train_tokens
        # writer.add_scalar('test/loss', test_loss, epoch)
        # writer.add_scalar('test/acc', test_acc, epoch)
        return

    test_accs = []
    best_val_acc_loc = 0
    labeled_train_iter, unlabeled_train_iter, mlm_train_iter = None, None, None
    # Start training
    for epoch in range(args.epochs):
        if args.resume and args.resume != "None":
            resume_filename = '{}_ep{}.ckpt'.format(args.resume, epoch)
            assert os.path.isfile(resume_filename), "=> no checkpoint found at '{}'".format(resume_filename)
            logger.info("=> loading checkpoint '{}'".format(resume_filename))
            checkpoint = torch.load(resume_filename)
            # epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            val_loss, val_acc = validate(val_loader, model, criterion, epoch, mode='Valid Stats')
            logger.info("epoch {}, val acc {}, val_loss {}".format(epoch, val_acc, val_loss))

            if val_acc >= best_acc:
                best_acc = val_acc
                best_val_acc_loc = epoch
                # test_loss, test_acc = validate(test_loader, model, criterion, epoch, mode='Test Stats ')
                # test_accs.append(test_acc)
                # logger.info("epoch {}, test acc {},test loss {}".format(epoch, test_acc, test_loss))

            logger.info('Best acc: {}'.format(best_acc))
            logger.info('Test acc: {}'.format(test_accs))
        else:
            mlm_set, labeled_train_iter, unlabeled_train_iter, mlm_train_iter = train(labeled_trainloader,
                                                                                      labeled_train_iter,
                                                                                      unlabeled_trainloader,
                                                                                      unlabeled_train_iter, mlm_set,
                                                                                      mlm_train_iter, model, optimizer,
                                                                                      optimizer_mlm, scheduler,
                                                                                      scheduler_mlm, train_criterion,
                                                                                      epoch, n_labels, args.train_aug)

            # save model
            ckpt = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'attention': mlm_set.attention
            }
            # filename = 'checkpoint.conditionalS4l_ep{}.ckpt'.format(epoch)
            # checkpoint_path = os.path.join('./ckpt/conditionalS4l/', filename)
            # filename = 'checkpoint.conditionalS4lNoAttention_ep{}.ckpt'.format(epoch)
            # checkpoint_path = os.path.join('./ckpt/conditionalS4lNoAttention/', filename)
            filename = 'checkpoint.{}_ep{}.ckpt'.format(run_context.exp_name, epoch)
            checkpoint_path = os.path.join('./ckpt/conditionalS4l', filename)
            torch.save(ckpt, checkpoint_path)

    logger.info("Finished training!")
    logger.info('Best val_acc: {} at {}'.format(best_acc, best_val_acc_loc))
    logger.info('Test acc: {}'.format(test_accs))


def train(labeled_trainloader, labeled_train_iter, unlabeled_trainloader, unlabeled_train_iter, mlm_set, mlm_train_iter, model, optimizer, optimizer_mlm, scheduler,
          scheduler_mlm, criterion, epoch, n_labels, train_aug=False):
    if labeled_train_iter is None:
        labeled_train_iter = iter(labeled_trainloader)
    if unlabeled_train_iter is None:
        unlabeled_train_iter = iter(unlabeled_trainloader)
    if mlm_train_iter is None:
        mlm_loader = Data.DataLoader(dataset=mlm_set, batch_size=args.mlm_batch_size, shuffle=True, drop_last=True)
        mlm_train_iter = iter(mlm_loader)
    model.train()

    global total_steps
    global flag
    global attention
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
                inputs_x, targets_x, inputs_x_length, idx_x = labeled_train_iter.next()
            except:
                labeled_train_iter = iter(labeled_trainloader)
                inputs_x, targets_x, inputs_x_length, idx_x = labeled_train_iter.next()
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
                                                length_u2, length_ori), idx_u, _ = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2, inputs_ori), (length_u,
                                                length_u2, length_ori), idx_u, _ = unlabeled_train_iter.next()
        data_time += time.time() - end
        end = time.time()

        batch_size = inputs_x.size(0)
        batch_size_2 = inputs_ori.size(0)
        targets_x = torch.zeros(batch_size, n_labels).scatter_(1, targets_x.view(-1, 1), 1)

        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
        inputs_u = inputs_u.cuda()
        inputs_u2 = inputs_u2.cuda()
        inputs_ori = inputs_ori.cuda()

        mask = []

        with torch.no_grad():
            # Predict labels for unlabeled data.
            if 'yahoo_answers' in args.data_path:
                pl_weight = pl_gen_weight['yahoo']
            else:
                raise LookupError
            _inputs = [inputs_u, inputs_u2, inputs_ori]
            p = 0
            for idx, cur_weight in enumerate(pl_weight):
                if cur_weight > 0:
                    cur_output = model(_inputs[idx])
                    p += cur_weight * torch.softmax(cur_output, dim=1)
            p /= sum(pl_weight)
            # outputs_u = model(inputs_u)
            # outputs_u2 = model(inputs_u2)
            # outputs_ori = model(inputs_ori)
            # p = (0 * torch.softmax(outputs_u, dim=1) + 0 * torch.softmax(outputs_u2,
            #                                                              dim=1) + 1 * torch.softmax(outputs_ori,
            #                                                                                         dim=1)) / 2
            # Do a sharpen here.
            pt = p ** (1 / args.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

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
                all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2, inputs_ori], dim=0)
                all_lengths = torch.cat([inputs_x_length, length_u, length_u2, length_ori], dim=0)
                all_targets = torch.cat([targets_x, targets_u, targets_u, targets_u], dim=0)
            else:
                all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2, inputs_ori, inputs_ori], dim=0)
                all_lengths = torch.cat([inputs_x_length, length_u, length_u2, length_ori, length_ori], dim=0)
                all_targets = torch.cat([targets_x, targets_u, targets_u, targets_u, targets_u], dim=0)
        else:
            all_inputs = torch.cat([inputs_x, inputs_x_aug, inputs_u, inputs_u2, inputs_ori], dim=0)
            all_lengths = torch.cat([inputs_x_length, inputs_x_length, length_u, length_u2, length_ori], dim=0)
            all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u, targets_u], dim=0)

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
            # input_a_sp = torch.split(input_a, 6, dim=0)
            # input_b_sp = torch.split(input_b, 6, dim=0)
            # logits = list()
            # for _input_a, _input_b in zip(input_a_sp, input_b_sp):
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
                        logger.info(256 - 1 - length1 - length2, idx2, length2, idx1, length1)

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

        # mlm training
        attention_all = retrieval_attention.get_attention(model, torch.cat([inputs_x, inputs_u], 0))
        attention[idx_x] = attention_all[:inputs_x.size()[0]]
        attention[idx_u] = attention_all[inputs_x.size()[0]:]
        # attention[idx_x] = get_attention(inputs_x)
        # attention[idx_u] = get_attention(inputs_u)
        # if total_steps % 10 == 0:
        if total_steps % args.val_iteration == 0:
            mlm_set.update_attention(attention)
        try:
            inputs_mlm, mask_mlm, targets_mlm, idx_mlm = mlm_train_iter.next()
        except:
            mlm_loader = Data.DataLoader(dataset=mlm_set, batch_size=args.mlm_batch_size, shuffle=True, drop_last=True)
            mlm_train_iter = iter(mlm_loader)
            inputs_mlm, mask_mlm, targets_mlm, idx_mlm = mlm_train_iter.next()

        inputs_mlm, targets_mlm, mask_mlm = inputs_mlm.cuda(), targets_mlm.cuda(), mask_mlm.cuda()
        mlm_prediction_scores = model(inputs_mlm, mlm=True, attention_mask=mask_mlm)
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        masked_lm_loss = loss_fct(mlm_prediction_scores.view(-1, vocab_size), targets_mlm.view(-1))
        optimizer_mlm.zero_grad()
        masked_lm_loss.backward()
        optimizer_mlm.step()
        scheduler_mlm.step()  # Update learning rate schedule

        # print(batch_idx)
        if batch_idx % 100 == 0:
            # logger.info("epoch {}, step {}, loss {}, Lx {}, Lu {}, Lu2 {}, mlm {}".format(
            #     epoch, batch_idx, loss.item(), Lx.item(), Lu.item(), Lu2.item(), masked_lm_loss.item()))
            # logger.info("epoch {}, step {}, loss {}, Lx {}, Lu {}, Lu2 {}".format(
            #     epoch, batch_idx, loss.item(), Lx.item(), Lu.item(), Lu2.item()))
            logger.info(
                "epoch {}, step {}, loss {}, Lx {}, Lu {}, mlm {}".format(epoch, batch_idx, loss.item(), Lx.item(),
                                                                          Lu.item(), masked_lm_loss.item()))
            # logger.info(
            #     "epoch {}, step {}, loss {}, Lx {}, Lu {}".format(epoch, batch_idx, loss.item(), Lx.item(),
            #                                                               Lu.item()))
        train_time += time.time() - end
        end = time.time()
    logger.info('---> train time: data {}, infer {}'.format(data_time, train_time))
    return mlm_set, labeled_train_iter, unlabeled_train_iter, mlm_train_iter


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

        for batch_idx, (inputs, targets, length, idx) in enumerate(valloader):
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
            # Lu2 = torch.mean(torch.clamp(torch.sum(-F.softmax(outputs_u_2, dim=1)
            #                                        * F.log_softmax(outputs_u_2, dim=1), dim=1) - args.margin, min=0))
            if outputs_u_2 is None:
                Lu2 = 0
            else:
                Lu2 = torch.mean(torch.clamp(torch.sum(-F.softmax(outputs_u_2, dim=1)
                                                       * F.log_softmax(outputs_u_2, dim=1), dim=1) - args.margin,
                                             min=0))

        elif args.mix_method == 2:
            if mixed == 0:
                Lx = - torch.mean(torch.sum(F.logsigmoid(outputs_x) * targets_x, dim=1))
                probs_u = torch.softmax(outputs_u, dim=1)
                Lu = F.kl_div(probs_u.log(), targets_u, None, None, 'batchmean')
                Lu2 = torch.mean(torch.clamp(args.margin - torch.sum(
                    F.softmax(outputs_u_2, dim=1) * F.softmax(outputs_u_2, dim=1), dim=1), min=0))
            else:
                Lx = - torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
                probs_u = torch.softmax(outputs_u, dim=1)
                Lu = F.kl_div(probs_u.log(), targets_u, None, None, 'batchmean')

                Lu2 = torch.mean(torch.clamp(args.margin - torch.sum(
                    F.softmax(outputs_u, dim=1) * F.softmax(outputs_u, dim=1), dim=1), min=0))

        return Lx, Lu, args.lambda_u * linear_rampup(epoch), Lu2, args.lambda_u_hinge * linear_rampup(epoch)


if __name__ == '__main__':
    main()
