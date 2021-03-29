import argparse
import os
import random
import math
import time
from pathlib import Path
import json
from tempfile import TemporaryDirectory
from collections import namedtuple
import logging
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from pytorch_transformers import *
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, RandomSampler

from read_data import *
from s4l_bert import S4LBert

parser = argparse.ArgumentParser(description='PyTorch S4L')

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

# MLM task
parser.add_argument('--pregenerated_data', type=Path, required=True)
parser.add_argument("--mlm_batch_size", default=32, type=int, help="Total batch size for training.")
parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")
parser.add_argument('--lrmlm', '--learning-rate-mlm', default=3e-5, type=float,
                    metavar='LR', help='initial learning rate for models')
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--reduce_memory", action="store_true",
                    help="Store training data as on-disc memmaps to massively reduce memory usage")

args = parser.parse_args()

assert args.pregenerated_data.is_dir(), \
    "--pregenerated_data should point to the folder of files made by pregenerate_training_data.py!"

samples_per_epoch = []
for i in range(args.epochs):
    epoch_file = args.pregenerated_data / f"epoch_{i}.json"
    metrics_file = args.pregenerated_data / f"epoch_{i}_metrics.json"
    if epoch_file.is_file() and metrics_file.is_file():
        metrics = json.loads(metrics_file.read_text())
        samples_per_epoch.append(metrics['num_training_examples'])
    else:
        if i == 0:
            exit("No training data was found!")
        print(f"Warning! There are fewer epochs of pregenerated data ({i}) than training epochs ({args.epochs}).")
        print("This script will loop over the available data, but training diversity may be negatively impacted.")
        num_data_epochs = i
        break
else:
    num_data_epochs = args.epochs

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("gpu num: ", n_gpu)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

num_train_optimization_steps = int(args.val_iteration * args.epochs)

best_acc = 0

InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids lm_label_ids is_next")

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)


def convert_example_to_features(example, tokenizer, max_seq_length):
    tokens = example["tokens"]
    segment_ids = example["segment_ids"]
    is_random_next = example["is_random_next"]
    masked_lm_positions = example["masked_lm_positions"]
    masked_lm_labels = example["masked_lm_labels"]

    assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

    input_array = np.zeros(max_seq_length, dtype=np.int)
    input_array[:len(input_ids)] = input_ids

    mask_array = np.zeros(max_seq_length, dtype=np.bool)
    mask_array[:len(input_ids)] = 1

    segment_array = np.zeros(max_seq_length, dtype=np.bool)
    segment_array[:len(segment_ids)] = segment_ids

    lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
    lm_label_array[masked_lm_positions] = masked_label_ids

    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             segment_ids=segment_array,
                             lm_label_ids=lm_label_array,
                             is_next=is_random_next)
    return features


class PregeneratedDataset(Dataset):
    def __init__(self, training_path, epoch, tokenizer, num_data_epochs, reduce_memory=False):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.data_epoch = epoch % num_data_epochs
        data_file = training_path / f"epoch_{self.data_epoch}.json"
        metrics_file = training_path / f"epoch_{self.data_epoch}_metrics.json"
        assert data_file.is_file() and metrics_file.is_file()
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        seq_len = metrics['max_seq_len']
        self.temp_dir = None
        self.working_dir = None
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            input_ids = np.memmap(filename=self.working_dir / 'input_ids.memmap',
                                  mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
            input_masks = np.memmap(filename=self.working_dir / 'input_masks.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            segment_ids = np.memmap(filename=self.working_dir / 'segment_ids.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            lm_label_ids = np.memmap(filename=self.working_dir / 'lm_label_ids.memmap',
                                     shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            lm_label_ids[:] = -1
            is_nexts = np.memmap(filename=self.working_dir / 'is_nexts.memmap',
                                 shape=(num_samples,), mode='w+', dtype=np.bool)
        else:
            input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            segment_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
            is_nexts = np.zeros(shape=(num_samples,), dtype=np.bool)
        logging.info(f"Loading training examples for epoch {epoch}")
        with data_file.open() as f:
            for i, line in enumerate(tqdm(f, total=num_samples, desc="Training examples")):
                line = line.strip()
                example = json.loads(line)
                features = convert_example_to_features(example, tokenizer, seq_len)
                input_ids[i] = features.input_ids
                segment_ids[i] = features.segment_ids
                input_masks[i] = features.input_mask
                lm_label_ids[i] = features.lm_label_ids
                is_nexts[i] = features.is_next
        assert i == num_samples - 1  # Assert that the sample count metric was true
        logging.info("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids
        self.is_nexts = is_nexts

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.segment_ids[item].astype(np.int64)),
                torch.tensor(self.lm_label_ids[item].astype(np.int64)),
                torch.tensor(self.is_nexts[item].astype(np.int64)))


def main():
    global best_acc
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # labeled_trainloader = Data.DataLoader(
    #     dataset=train_labeled_set, batch_size=args.batch_size, shuffle=True)
    # val_loader = Data.DataLoader(
    #     dataset=val_set, batch_size=512, shuffle=False)
    # test_loader = Data.DataLoader(
    #     dataset=test_set, batch_size=512, shuffle=False)

    model = S4LBert().cuda()
    model = nn.DataParallel(model)
    # optimizer = AdamW(
    #     [
    #         {"params": model.module.bert.parameters(), "lr": args.lrmain},
    #         {"params": model.module.linear.parameters(), "lr": args.lrlast},
    #     ])
    # MLM optim
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer_mlm = AdamW(optimizer_grouped_parameters, lr=args.lrmlm, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer_mlm, warmup_steps=args.warmup_steps,
                                     t_total=num_train_optimization_steps)

    # criterion = nn.CrossEntropyLoss()

    test_accs = []
    mlm_epoch = 0
    mlm_dataset = PregeneratedDataset(epoch=mlm_epoch, training_path=args.pregenerated_data, tokenizer=tokenizer,
                                      num_data_epochs=num_data_epochs, reduce_memory=args.reduce_memory)
    mlm_sampler = RandomSampler(mlm_dataset)
    mlm_dataloader = DataLoader(mlm_dataset, sampler=mlm_sampler, batch_size=args.mlm_batch_size)
    mlm_iter = iter(mlm_dataloader)

    for epoch in range(args.epochs):
        mlm_epoch, mlm_iter = train(mlm_iter, model, optimizer_mlm, scheduler, epoch, mlm_epoch, tokenizer)

        # save model
        ckpt = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
        }
        filename = 'checkpoint.pretrain_ep{}.ckpt'.format(epoch)
        checkpoint_path = os.path.join('./ckpt/pretrain/', filename)
        torch.save(ckpt, checkpoint_path)


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

        acc_total = correct / total_sample
        loss_total = loss_total / total_sample

    return loss_total, acc_total


def train(mlm_iter, model, optimizer_mlm, scheduler, epoch, mlm_epoch, tokenizer):
    model.train()

    data_time = 0
    train_time = 0
    end = time.time()
    for batch_idx in range(args.val_iteration):
        try:
            mlm_batch = mlm_iter.next()
            mlm_batch = tuple(t.to(device) for t in mlm_batch)
            input_ids, input_mask, segment_ids, lm_label_ids, is_next = mlm_batch
        except:
            mlm_epoch += 1
            mlm_dataset = PregeneratedDataset(epoch=mlm_epoch, training_path=args.pregenerated_data,
                                              tokenizer=tokenizer,
                                              num_data_epochs=num_data_epochs, reduce_memory=args.reduce_memory)
            mlm_sampler = RandomSampler(mlm_dataset)
            mlm_dataloader = DataLoader(mlm_dataset, sampler=mlm_sampler, batch_size=args.mlm_batch_size)
            mlm_iter = iter(mlm_dataloader)
            mlm_batch = mlm_iter.next()
            mlm_batch = tuple(t.to(device) for t in mlm_batch)
            input_ids, input_mask, segment_ids, lm_label_ids, is_next = mlm_batch

        # for batch_idx, (inputs, targets, length) in enumerate(labeled_trainloader):
        data_time += time.time() - end
        end = time.time()

        # mlm training
        mlm_loss = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, masked_lm_labels=lm_label_ids,
                        next_sentence_label=is_next)
        if n_gpu > 1:
            mlm_loss = mlm_loss.mean()  # mean() to average on multi-gpu.
        mlm_loss.backward()
        optimizer_mlm.zero_grad()
        optimizer_mlm.step()
        scheduler.step()  # Update learning rate schedule

        print('epoch {}, step {}, mlm loss {}'.format(epoch, batch_idx, mlm_loss.item()))
        train_time += time.time() - end
        end = time.time()
    print('train time: data {}, infer {}'.format(data_time, train_time))
    return mlm_epoch, mlm_iter


if __name__ == '__main__':
    main()
