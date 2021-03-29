import torch
import torch.nn as nn
from pytorch_transformers import *
import ipdb


class ClassificationBert(nn.Module):
    def __init__(self, num_labels=2, use_gap=True, require_attention=False, use_cls_sep=False):
        super(ClassificationBert, self).__init__()
        # Load pre-trained bert model
        # self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.require_attention = require_attention
        config = BertConfig.from_pretrained(
            './cache/pytorch_transformers/4dad0251492946e18ac39290fcfe91b89d370fee250efe9521476438fe8ca185.7156163d5fdc189c3016baca0775ffce230789d7fa2a42ef516483e4ca884517',
            cache_dir=None, force_download=False, output_hidden_states=require_attention,
            output_attentions=require_attention)
        self.bert = BertModel.from_pretrained('./cache/pytorch_transformers/aa1ef1aede4482d0dbcd4d52baad8ae300e60902e88fcb0bebdec09afd232066.36ca03ab34a1a5d5fa7bc3d03d55c4fa650fed07220e2eeebc06ce58d0e9a157',
                                              config=config)
        self.use_gap = use_gap
        self.use_cls_sep = use_cls_sep
        # if use_gap:
        # for global avg pooling
        self.classifier = nn.Sequential(nn.Linear(self.bert.config.hidden_size, 128),
                                        nn.Tanh(),
                                        nn.Linear(128, num_labels))
        # else:
        #     # for [CLS] token
        #     self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        #     self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

        self.gradients = None
        self.activations = None
        self.attentions = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x, length=None, attention_mask=None, token_type_ids=None, output_attention=False):
        # Encode input text
        if length is not None:
            len_range = torch.arange(x.size()[-1], device=x.device, dtype=length.dtype).expand(*x.size())
            attention_mask = (len_range < length.unsqueeze(-1)).long()
            if self.require_attention:
                if output_attention:
                    all_hidden, pooler, _, attentions = self.bert(x, attention_mask=attention_mask)
                    self.attentions = attentions
                else:
                    all_hidden, pooler, _, _ = self.bert(x, attention_mask=attention_mask)
            else:
                all_hidden, pooler = self.bert(x, attention_mask=attention_mask)
        else:
            if self.require_attention:
                if output_attention:
                    all_hidden, pooler, _, attentions = self.bert(x, attention_mask=attention_mask, token_type_ids=token_type_ids)
                    self.attentions = attentions
                else:
                    all_hidden, pooler, _, _ = self.bert(x, attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                all_hidden, pooler = self.bert(x, attention_mask=attention_mask, token_type_ids=token_type_ids)

        if self.use_gap:
            # import ipdb; ipdb.set_trace()
            # self.activations = all_hidden
            # h = all_hidden.register_hook(self.activations_hook)
            if not self.use_cls_sep:
                # ======== gap on feature only (assume cls&sep are used) ==========
                len_range = torch.arange(x.size()[-1], device=x.device, dtype=length.dtype).expand(*x.size())
                mask = (len_range < (length - 1).unsqueeze(-1)).long()
                mask[:, 0] = 0
                masked_out = all_hidden * mask.unsqueeze(-1).expand(*all_hidden.size())
                pooled_output = torch.sum(masked_out, 1) / (length - 2).unsqueeze(-1)
            else:
                # ======== gap on all output sequence ========
                pooled_output = torch.mean(all_hidden, 1)
        else:
            # self.activations = pooler
            # h = pooler.register_hook(self.activations_hook)
            # pooled_output = self.dropout(pooler)
            # pooled_output = pooler
            pooled_output = all_hidden[:, 0, :]
        predict = self.classifier(pooled_output)
        if output_attention:
            return predict, self.attentions
        else:
            return predict

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self):
        return self.activations
