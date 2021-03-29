import argparse

import torch
import torch.nn as nn
import torchtext
from torchtext import data
from torchtext.datasets import text_classification
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import os
from pytorch_transformers import *
from transformers import BertTokenizer, BertConfig, BertModel, BertForMaskedLM, BertForSequenceClassification, BertPreTrainedModel
from transformers.modeling_bert import BertEncoder, BertPooler

parser = argparse.ArgumentParser(description='PyTorch Base Models')
parser.add_argument('--model', type=str, default='bert-base-uncased', help='pretrained model')

parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch-size', default=4, type=int, metavar='N', help='train batchsize')
parser.add_argument('--batch-size-test', default=64, type=int, metavar='N', help='test batchsize')

parser.add_argument('--opt-name', default='Adam', type=str)
parser.add_argument('--lrmain', '--learning-rate-bert', default=2e-5, type=float,
                    metavar='LR', help='initial learning rate for bert')
parser.add_argument('--lrlast', '--learning-rate-model', default=-1, type=float,
                    metavar='LR', help='initial learning rate for models')

parser.add_argument('--ds_name', type=str, default='AG_NEWs', help='dataset names')
parser.add_argument('--data-path', type=str, default='yahoo_answers_csv/', help='path to data folders')
parser.add_argument('--n-labeled', type=int, default=5, help='Number of labeled data')
parser.add_argument('--n-valid', type=int, default=2000, help='Number of data for validation')
parser.add_argument('--save-path', type=str, default='./model/', help='path to data folders')

parser.add_argument('--max-seq-len', default=0, type=int, help='pre-defined max seqence length')
parser.add_argument('--add_prob', action="store_true", default=False, help='adding probing words')
parser.add_argument('--prob_word_type', default='NEWS', type=str, help="['NEWS', 'CLASS', 'ClassWise', 'LabList', 'SimList']")
parser.add_argument('--avg_prob_at', default='None', type=str, help='input, output, None')
parser.add_argument('--add-cls-sep', action="store_true", default=False)
# parser.add_argument('--classify-with-cls', action="store_true", default=False)
# parser.add_argument('--use-cls-type', default='normal', type=str, help='qy: out[0][:, 0, :]; normal: out[1]')
parser.add_argument('--classify-with', default='gap', type=str, help='gap, clsQY, clsNrml, avgPrb, gapFeaOnly, gapFeaPrb')

parser.add_argument('--gpu', default='0,1,2,3', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', default=514, type=int, metavar='N', help='random seed')

parser.add_argument('--specific_name', default=None, type=str, help='the specific name for the output file')
args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random_seed = args.seed
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

pretrained_bert = args.model
tokenizer = BertTokenizer.from_pretrained(pretrained_bert, do_lower_case=True)
# bert_cache_dir = '~/.pytorch_pretrained_bert/'

save_folder = args.save_path

# 'NEWS', 'CLASS', 'ClassWise', 'LabList', 'SimList'
EXTRA_TOKEN = args.prob_word_type
# EXTRA_TOKEN_LIST = ['World', 'Sports', 'Business', 'Sci/Tech']
# EXTRA_TOKEN_LIST=[]


def construct_file_name(pre_name):
    pre_name += '_nL' + str(args.n_labeled)
    pre_name += '_bs' + str(args.batch_size)
    pre_name += '_opt' + args.opt_name
    pre_name += '_lr' + str(args.lrmain) + '_' + str(args.lrlast)
    pre_name += '_ep' + str(args.epochs)
    pre_name += '_wClsSep' if args.add_cls_sep else '_noClsSep'
    pre_name += '_wPrbWrd' + EXTRA_TOKEN if args.add_prob else '_noPrbWrd'
    pre_name += '_avgAt' + args.avg_prob_at if args.add_prob and EXTRA_TOKEN in ['LabList', 'SimList'] else ''
    pre_name += '_maxSeqLen' + str(args.max_seq_len)
    # pre_name += '_cls' + args.use_cls_type if args.classify_with_cls else '_wGAP'
    pre_name += '_clsW' + args.classify_with
    pre_name += '_seed' + str(args.seed)
    pre_name += args.specific_name if args.specific_name is not None else ''

    pre_name += '.pt'
    return pre_name


saved_model_name = construct_file_name('model')
saved_metrics_name = construct_file_name('metrics')

Dataset_name = args.ds_name
if Dataset_name == 'AG_NEWs':
    train_raw = pd.read_csv('./data/ag_news_csv/train.csv', names=['Class Index', 'Title', 'Description'])
    test_raw = pd.read_csv('./data/ag_news_csv/test.csv', names=['Class Index', 'Title', 'Description'])
    LABEL_LIST = [['world'], ['sports'], ['business'], ['science', 'technology']]
    SIMILAR_LIST = [['global'], ['footable'], ['trade'], ['electronic', 'device']]

    def translate_data(raw_data):
        title = raw_data['Title'].tolist()
        des = raw_data['Description'].tolist()
        label = raw_data['Class Index'].tolist()
        return {
            'title': title,
            'des': des,
            'label': label
        }

    train_data = translate_data(train_raw)
    test_data = translate_data(test_raw)
else:
    raise LookupError

args.prob_len = len(sum(LABEL_LIST, [])) if args.prob_word_type in ['LabList', 'SimList'] else 1
all_classes = np.unique(test_data['label'])
num_labels = len(all_classes)
train_idx_list = []
for y in all_classes:
    train_idx_list.append(np.squeeze(np.argwhere(train_data['label'] == y)))
num_all_per_class = len(train_idx_list[0])
# print(num_all_per_class)
indices = list(range(num_all_per_class))

# validation_split = 0.01
num_train_per_class = args.n_labeled
num_val_per_class = args.n_valid
training_batch_size = args.batch_size
testing_batch_size = args.batch_size_test
num_epochs = args.epochs

train_indices, val_indices = [], []
for i in range(num_labels):
    np.random.shuffle(train_idx_list[i])
    train_indices.extend(train_idx_list[i][:num_train_per_class].tolist())
    #     print(train_idx_list[i][:num_train_per_class])
    val_indices.extend(train_idx_list[i][num_train_per_class:num_train_per_class + num_val_per_class].tolist())
#    print(train_idx_list[i][num_train_per_class:num_train_per_class+num_val_per_class])
#    print("----")

# label = 0
# for i in range(len(train_indices)):
#     if i % num_train_per_class == 0:
#         label = label +1
#     idx = train_indices[i]
#     if train_data['label'][idx] != label:
#         print(idx)
#         print(label)
#         print(train_data['label'][idx])

# label = 0
# for i in range(len(val_indices)):
#     if i % num_val_per_class == 0:
#         label = label +1
#     idx = val_indices[i]
#     if train_data['label'][idx] != label:
#         print(idx)
#         print(label)
#         print(train_data['label'][idx])
test_indices = list(range(len(test_data['label'])))


def _del_by_idx(array_list, idx, axis):
    '''
        Delete the specified index for each array in the array_lists

        @params: array_list: list of np arrays
        @params: idx: list of int
        @params: axis: int

        @return: res: tuple of pruned np arrays
    '''
    if type(array_list) is not list:
        array_list = [array_list]

    # modified to perform operations in place
    for i, array in enumerate(array_list):
        array_list[i] = np.delete(array, idx, axis)

    if len(array_list) == 1:
        return array_list[0]
    else:
        return array_list


def process_data(data, Dataset_name, tokenizer, Extra_label=False, EXTRA_TOKEN=None, max_len_def=0, filter_indices=None):
    if filter_indices is None:
        filter_indices = []
    print("add extra label: ", Extra_label)
    print("extra label: ", EXTRA_TOKEN)
    assert max_len_def == 0, 'no need to pre-define max seq len!!!'
    if Dataset_name == 'AG_NEWs':
        title = data['title']
        des = data['des']
        label = data['label']
        total_data = len(title)
        token_id = []
        max_len = 0
        for i in range(len(des)):
            comb_text = title[i] + " " + des[i]
            token_id.append(tokenizer.encode(comb_text, add_special_tokens=args.add_cls_sep))
            max_len = max(len(token_id[i]), max_len)
        if max_len_def > 0:
            max_len = max_len_def
            pre_def_len = True
        else:
            pre_def_len = False

        del_idx = []
        ex_len = 0
        # convert each token to its corresponding id
        if Extra_label:
            print("add extra label: ", EXTRA_TOKEN)
            ex_token = EXTRA_TOKEN
            if ex_token == 'NEWS' or ex_token == 'CLASS':
                text_label = np.zeros([2], dtype=np.int64) if args.add_cls_sep else np.zeros([1], dtype=np.int64)
                text_label[0] = tokenizer.encode(ex_token, add_special_tokens=False)[0]
                if args.add_cls_sep:
                    text_label[1] = 102
                ex_len = len(text_label)
            elif ex_token == 'ClassWise' or ex_token == 'LabList':
                text_label = list()
                ex_len = 0
                for cls_name in LABEL_LIST:
                    cur_cls_token_id = tokenizer.encode(cls_name, add_special_tokens=False)
                    if ex_token == 'ClassWise':
                        if args.add_cls_sep:
                            cur_cls_token_id.extend([102])
                        ex_len = max(len(cur_cls_token_id), ex_len)
                    else:
                        ex_len += len(cur_cls_token_id)
                    text_label.append(cur_cls_token_id)
                if ex_token == 'LabList':
                    if args.add_cls_sep:
                        text_label.append([102])
                        ex_len += 1
                    text_label = sum(text_label, [])
            elif ex_token == 'SimList':
                text_label = list()
                ex_len = 0
                for cls_name in SIMILAR_LIST:
                    cur_cls_token_id = tokenizer.encode(cls_name, add_special_tokens=False)
                    text_label.append(cur_cls_token_id)
                    ex_len += len(cur_cls_token_id)
                if args.add_cls_sep:
                    text_label.append([102])
                    ex_len += 1
                text_label = sum(text_label, [])
            else:
                raise LookupError

        if not pre_def_len:
            max_len = max_len + ex_len
        text_len = np.array([len(tok) for tok in token_id])
        text = np.zeros((total_data, max_len), dtype=np.int64)

        if pre_def_len:
            for i in range(total_data):
                cur_text_len = text_len[i]
                if Extra_label:
                    if cur_text_len + ex_len >= max_len:
                        text[i, :(max_len - ex_len)] = token_id[i][:(max_len - ex_len)]
                        text[i, - ex_len:] = text_label
                        text_len[i] = max_len
                    else:
                        text[i, :cur_text_len] = token_id[i]
                        text[i, cur_text_len:cur_text_len + ex_len] = text_label
                        text_len[i] = text_len[i] + ex_len
                else:
                    if cur_text_len >= max_len:
                        text[i, :max_len] = token_id[i][:max_len]
                        text_len[i] = max_len
                    else:
                        text[i, :cur_text_len] = token_id[i]

                # filter out document with only special tokens
                # unk (100), cls (101), sep (102), pad (0)
                if np.max(text[i]) < 103:
                    del_idx.append(i)
        else:
            for i in range(total_data):
                text[i, :len(token_id[i])] = token_id[i]
                if Extra_label:
                    if EXTRA_TOKEN in ['NEWS', 'CLASS', 'LabList', 'SimList']:
                        if i in filter_indices:
                            continue
                        else:
                            text[i, len(token_id[i]):len(token_id[i]) + ex_len] = text_label
                            text_len[i] = text_len[i] + ex_len
                    elif EXTRA_TOKEN == 'ClassWise':
                        if i in filter_indices:
                            continue
                        else:
                            cur_sample_ex_len = len(text_label[label[i]-1])
                            text[i, len(token_id[i]):len(token_id[i]) + cur_sample_ex_len] = text_label[label[i]-1]
                            text_len[i] = text_len[i] + cur_sample_ex_len
                    else:
                        raise LookupError

                # filter out document with only special tokens
                # unk (100), cls (101), sep (102), pad (0)
                if np.max(text[i]) < 103:
                    del_idx.append(i)
        #         print(text[0])
        #         print(text[0])
    else:
        text_len = None
        text = None
        label = None
        del_idx = None
    #     print(max_len)
    text_len, text, label = _del_by_idx([text_len, text, label], del_idx, 0)
    new_data = {
        'token_id': text,
        'token_len': text_len,
        'label': label
    }
    return new_data


def process_data2(data, Dataset_name, tokenizer, Extra_label=False, EXTRA_TOKEN_LIST=None):
    print("add extra label: ", Extra_label)
    print("extra label: ", EXTRA_TOKEN_LIST)
    if Dataset_name == 'AG_NEWs':
        title = data['Title'].tolist()
        des = data['Description'].tolist()
        label = data['Class Index'].tolist()
        token_id = []
        max_len = 0
        for i in range(len(des)):
            comb_text = title[i] + " " + des[i]
            token_id.append(tokenizer.encode(comb_text, add_special_tokens=True))
            max_len = max(len(token_id[i]), max_len)

        del_idx = []
        ex_len = 0
        # convert each token to its corresponding id
        EMBEDDING_EXTRA_TOKEN_LIST = []
        if Extra_label == True:
            #             print("add extra label: ", EXTRA_TOKEN)
            for ex_token in EXTRA_TOKEN_LIST:
                #                 text_label = []
                #                 text_label.append( tokenizer.encode(ex_token+'[SEP]',add_special_tokens=False))
                #                 print("token!!!")
                #                 print(tokenizer.encode(ex_token+'[SEP]',add_special_tokens=False))
                #                 text_label.append (102)
                text_label = tokenizer.encode(ex_token + '[SEP]', add_special_tokens=False)
                EMBEDDING_EXTRA_TOKEN_LIST.append(text_label)
                if len(text_label) > ex_len:
                    ex_len = len(text_label)

        max_len = max_len + ex_len
        text_len = np.array([len(tok) for tok in token_id])
        text = np.zeros((len(data), max_len), dtype=np.int64)

        print("=" * 20)
        print(EMBEDDING_EXTRA_TOKEN_LIST)
        for i in range(len(data)):
            text[i, :len(token_id[i])] = token_id[i]
            if Extra_label:
                #                 print(int(label(i))+1)

                text[i, len(token_id[i]):len(token_id[i]) + len(EMBEDDING_EXTRA_TOKEN_LIST[int(label[i]) - 1])] = \
                    EMBEDDING_EXTRA_TOKEN_LIST[int(label[i]) - 1]
                text_len[i] = text_len[i] + ex_len

            # filter out document with only special tokens
            # unk (100), cls (101), sep (102), pad (0)
            if np.max(text[i]) < 103:
                del_idx.append(i)
        #         print(text[0])
        print(text)
    else:
        text_len = None
        text = None
        label = None
        del_idx = None
    print(max_len)
    text_len, text, label = _del_by_idx([text_len, text, label], del_idx, 0)
    new_data = {
        'token_id': text,
        'token_len': text_len,
        'label': label
    }
    return new_data


train_data = process_data(train_data, Dataset_name, tokenizer, args.add_prob, EXTRA_TOKEN, max_len_def=args.max_seq_len,
                          filter_indices=val_indices if EXTRA_TOKEN == 'ClassWise' else None)
# train_data = process_data(train_data, Dataset_name, tokenizer, args.add_prob, EXTRA_TOKEN, max_len_def=args.max_seq_len,
#                           filter_indices=val_indices)
test_data = process_data(test_data, Dataset_name, tokenizer, False if EXTRA_TOKEN == 'ClassWise' else args.add_prob,
                         EXTRA_TOKEN, max_len_def=args.max_seq_len)


class myDataset(data.Dataset):
    def __init__(self, datasource):
        self.data = datasource['token_id']
        self.data_len = datasource['token_len']
        self.label = datasource['label']

    def __getitem__(self, index):
        data_ts = torch.tensor(self.data[index], dtype=torch.long)
        data_len_ts = torch.tensor(self.data_len[index], dtype=torch.long)
        label_ts = torch.tensor(self.label[index], dtype=torch.long) - 1
        return data_ts, data_len_ts, label_ts

    def __len__(self):
        return len(self.data)


train_dataset = myDataset(train_data)
test_dataset = myDataset(test_data)

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)
training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=training_batch_size, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(train_dataset, batch_size=testing_batch_size, sampler=valid_sampler)
testing_loader = torch.utils.data.DataLoader(test_dataset, batch_size=testing_batch_size, sampler=test_sampler)

BertLayerNorm = torch.nn.LayerNorm


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, text_len=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        assert args.add_cls_sep and args.add_prob and args.prob_word_type in ['LabList', 'SimList']
        probing_word_len = 5
        len_range = torch.arange(input_ids.size()[-1], device=input_ids.device, dtype=text_len.dtype).expand(
            *input_ids.size())
        mask1 = (len_range < (text_len-1).unsqueeze(-1)).long()
        mask2 = (len_range < (text_len-1-probing_word_len).unsqueeze(-1)).long()
        mask3 = ((mask1 + mask2) == 1).long()
        mask3 = mask3.unsqueeze(-1).repeat(1, 1, words_embeddings.size()[-1])
        prob_word_avg_embeding = torch.sum(words_embeddings * mask3, 1, keepdims=True) / 5
        prob_word_avg_embeding = prob_word_avg_embeding.repeat(1, input_ids.size()[1], 1)
        mask4 = (len_range == (text_len-1-probing_word_len).unsqueeze(-1)).long()
        mask4 = mask4.unsqueeze(-1).repeat(1, 1, words_embeddings.size()[-1])
        # unk (100), cls (101), sep (102), pad (0)
        all_sep = torch.ones(input_ids.size(), device=input_ids.device, dtype=input_ids.dtype) * 102
        all_sep_embeddings = self.word_embeddings(all_sep)
        mask5 = (len_range == (text_len-probing_word_len).unsqueeze(-1)).long()
        mask5 = mask5.unsqueeze(-1).repeat(1, 1, words_embeddings.size()[-1])
        all_pad = torch.zeros(input_ids.size(), device=input_ids.device, dtype=input_ids.dtype)
        all_pad_embeddings = self.word_embeddings(all_pad)
        mask2 = mask2.unsqueeze(-1).repeat(1, 1, words_embeddings.size()[-1])
        mask6 = 1 - (mask2 + mask4 + mask5)
        words_embeddings = words_embeddings * mask2 + prob_word_avg_embeding * mask4 + all_sep_embeddings * mask5 + all_pad_embeddings * mask6

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertModelAvgPrb(BertPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    def __init__(self, config):
        super(BertModelAvgPrb, self).__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, text_len=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids, text_len=text_len)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class TextModel(nn.Module):
    def __init__(self, config, num_labels, pretrained_model_name_or_path, cache_dir):
        super().__init__()
        if args.add_prob and args.avg_prob_at == 'input':
            self.bert = BertModelAvgPrb.from_pretrained(pretrained_model_name_or_path, cache_dir=cache_dir,
                                                        config=config)
        else:
            self.bert = BertModel.from_pretrained(pretrained_model_name_or_path, cache_dir=cache_dir, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, token_id, text_len, labels=None):
        len_range = torch.arange(token_id.size()[-1], device=token_id.device, dtype=text_len.dtype).expand(
            *token_id.size())
        mask1 = (len_range < text_len.unsqueeze(-1)).long()
        if args.add_prob and args.avg_prob_at == 'input':
            out = self.bert(token_id, attention_mask=mask1, text_len=text_len)
        else:
            out = self.bert(token_id, attention_mask=mask1)

        # gap, clsQY, clsNrml, avgPrb, gapFeaOnly, gapFeaPrb
        if args.classify_with == 'clsQY':
            output = out[0][:, 0, :]
        elif args.classify_with == 'clsNrml':
            output = out[1]
        elif args.classify_with == 'gap':
            output = torch.mean(out[0], dim=1)
        elif args.classify_with == 'gapFeaOnly':
            shift = 1 if args.add_cls_sep else 0
            shift += args.prob_len if args.add_prob else 0
            mask = (len_range < (text_len - shift).unsqueeze(-1)).long()
            if args.add_cls_sep:
                mask[:, 0] = 0
                shift += 1
            masked_out = out[0] * mask.unsqueeze(-1).repeat(1, 1, out[0].shape[-1])
            output = torch.sum(masked_out, 1) / (text_len - shift).unsqueeze(-1)
        # ============ filter CLS & SEP token's feature
        # mask2 = (len_range < (text_len - 1).unsqueeze(-1)).long()
        # mask2 = (len_range < (text_len - 5).unsqueeze(-1)).long()
        # mask2 = (len_range < (text_len - 6).unsqueeze(-1)).long()
        # mask2[:, 0] = 0
        # masked_out = out[0] * mask2.unsqueeze(-1).repeat(1, 1, out[0].shape[-1])
        # output = torch.sum(masked_out, 1) / (text_len-2).unsqueeze(-1)
        elif args.classify_with == 'avgPrb':
            import ipdb; ipdb.set_trace()
            assert args.add_prob
            sep_loc = 1 if args.add_cls_sep else 0
            prob_loc = args.prob_len + sep_loc
            if args.avg_prob_at == 'input':
                mask = (len_range == (text_len - prob_loc).unsqueeze(-1)).long()
                mask = mask.unsqueeze(-1).repeat(1, 1, out[0].size()[-1])
                output = torch.sum(out[0] * mask, 1)
            else:
                mask1 = (len_range < (text_len - sep_loc).unsqueeze(-1)).long()
                mask2 = (len_range < (text_len - prob_loc).unsqueeze(-1)).long()
                mask3 = ((mask1 + mask2) == 1).long()
                mask3 = mask3.unsqueeze(-1).repeat(1, 1, out[0].size()[-1])
                output = torch.sum(out[0] * mask3, 1)
        else:
            raise LookupError

        output = self.dropout(output)
        logits = self.classifier(output)
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits


# Save and Load Functions
def save_checkpoint(save_path, model, valid_acc):
    if save_path is None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_acc': valid_acc}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model):
    if load_path is None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_acc']


def save_metrics(save_path, train_loss_list, train_acc_list, valid_loss_list, valid_acc_list, global_steps_list):
    if save_path is None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'train_acc_list': train_acc_list,
                  'valid_loss_list': valid_loss_list,
                  'valid_acc_list': valid_acc_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):
    if load_path is None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    return state_dict['train_loss_list'], state_dict['train_acc_list'], state_dict['valid_loss_list'], state_dict[
        'valid_acc_list'], state_dict['global_steps_list']


def train(model, optimizer, train_loader=training_loader, valid_loader=validation_loader, num_epochs=num_epochs,
          eval_every=len(training_loader), file_path=save_folder, best_valid_acc=0.0):
    running_loss = 0.0
    valid_running_loss = 0.0
    train_acc = 0.0
    valid_acc = 0.0
    train_acc_list = []
    valid_acc_list = []
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []
    # training loop
    model.train()
    for epoch in range(num_epochs):
        for idx, data in enumerate(train_loader):
            #             print(data[0][0][:data[1][0]+1])
            #             print(data[1])
            #             print(data[2])
            token_id = data[0].to(device)
            token_len = data[1].to(device)
            label = data[2].to(device)
            loss, logits = model(token_id, token_len, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_acc += (logits.argmax(1) == label).sum().item() / training_batch_size
            # update running values
            running_loss += loss.item()
            global_step += 1
            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    # validation loop
                    for idx, val_data in enumerate(valid_loader):
                        token_id = val_data[0].to(device)
                        token_len = val_data[1].to(device)
                        label = val_data[2].to(device)
                        loss, logits = model(token_id, token_len, label)
                        valid_running_loss += loss.item()
                        valid_acc += (logits.argmax(1) == label).sum().item() / testing_batch_size
                # evaluation
                average_train_loss = running_loss / eval_every
                average_train_acc = train_acc / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                average_valid_acc = valid_acc / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                train_acc_list.append(average_train_acc)
                valid_acc_list.append(average_valid_acc)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0
                valid_running_loss = 0.0
                train_acc = 0.0
                valid_acc = 0.0
                model.train()

                # print progress
                print(
                    'Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Valid Loss: {:.4f}, Valid Acc: {:.4f},'
                        .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),
                                average_train_loss, average_train_acc, average_valid_loss, average_valid_acc))
                # checkpoint
                if best_valid_acc < average_valid_acc:
                    best_valid_acc = average_valid_acc
                    save_checkpoint(file_path + saved_model_name, model, best_valid_acc)
                    save_metrics(file_path + saved_metrics_name, train_loss_list, train_acc_list, valid_loss_list,
                                 valid_acc_list, global_steps_list)

    save_metrics(file_path + saved_metrics_name, train_loss_list, train_acc_list, valid_loss_list, valid_acc_list,
                 global_steps_list)

    print('Finished Training!')


config = BertConfig()
config.output_attentions = True
model = TextModel(config, num_labels, pretrained_bert, None).to(device)
if args.opt_name == 'Adam':
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lrmain)
    optimizer = torch.optim.Adam(
        [
            {"params": model.bert.parameters(), "lr": args.lrmain},
            {"params": model.classifier.parameters(), "lr": args.lrlast},
        ]
    )
else:
    optimizer = AdamW(
        [
            {"params": model.bert.parameters(), "lr": args.lrmain},
            {"params": model.classifier.parameters(), "lr": args.lrlast},
        ])

train(model=model, optimizer=optimizer)

# train_loss_list, train_acc_list, valid_loss_list, valid_acc_list, global_steps_list = load_metrics(save_folder + '/metrics.pt')
# plt.plot(global_steps_list, train_loss_list, label='Train')
# plt.plot(global_steps_list, valid_loss_list, label='Valid')
# plt.xlabel('Global Steps')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
