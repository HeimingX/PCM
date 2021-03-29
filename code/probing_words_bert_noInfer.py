import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from pytorch_transformers import *
from transformers.modeling_bert import BertEncoder, BertPooler

import global_file
import ipdb

BertLayerNorm = torch.nn.LayerNorm


class ProbingWordWeight(nn.Module):
    def __init__(self, in_channels: int):
        super(ProbingWordWeight, self).__init__()
        self.in_channels = in_channels
        # self.weight = nn.Parameter(torch.ones(in_channels))
        self.weight = nn.Parameter(torch.randn(in_channels))

    def forward(self, x, norm_func='softmax'):
        bs, seq_len, hidden_size = x.shape
        _weight = self.weight.unsqueeze(1).repeat(1, hidden_size)  # seq_len, hidden_size
        if norm_func == 'softmax':
            _weight = torch.softmax(_weight, 0)
        elif norm_func == 'reluNorm':
            _weight = nn.ReLU()(_weight)
            _weight /= torch.sum(_weight, 0, keepdim=True)
        else:
            raise LookupError
        _weight = _weight.unsqueeze(0).repeat(bs, 1, 1)
        res = _weight * x
        return res


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.weighted_prob = global_file.args.weighted_prob_word
        if self.weighted_prob:
            for idx in range(global_file.args.n_labels):
                cur_cls_probing_words_name = 'cls' + str(idx) + '_weight'
                cur_cls_probing_words_num = global_file.attended_token_num[idx]
                setattr(self, cur_cls_probing_words_name, ProbingWordWeight(cur_cls_probing_words_num))
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, probing_words_list=None):
        num_seq, seq_length = input_ids.size()
        words_embeddings = self.word_embeddings(input_ids)
        if probing_words_list is not None:
            probing_words_embedding = list()
            for cls_id in range(len(probing_words_list)):
                cur_cls_words_embeddings = self.word_embeddings(probing_words_list[cls_id])
                # print('cur_cls_words_embeddings shape1: {}'.format(cur_cls_words_embeddings.shape))
                if self.weighted_prob:
                    if cls_id == len(probing_words_list) - 1:
                        # sep token
                        cur_cls_avg_embedding = cur_cls_words_embeddings.mean(1, keepdim=True)
                    else:
                        cur_cls_probing_words_name = 'cls' + str(cls_id) + '_weight'
                        cur_cls_avg_embedding = getattr(self, cur_cls_probing_words_name)(cur_cls_words_embeddings,
                                                                                          norm_func=global_file.args.weighted_type)
                        cur_cls_avg_embedding = cur_cls_avg_embedding.sum(1, keepdim=True)
                elif global_file.args.parallel_prob:
                    if cls_id == len(probing_words_list) - 1:
                        # sep token
                        cur_cls_avg_embedding = cur_cls_words_embeddings.mean(1, keepdim=True)
                    else:
                        if global_file.args.dropout_prob > 0:
                            _, num_prob, hidden_size = cur_cls_words_embeddings.shape
                            random_tensor = torch.tensor(np.random.binomial(n=1, p=global_file.args.dropout_prob,
                                                                            size=(num_seq, num_prob))).cuda()
                            random_tensor = random_tensor.unsqueeze(-1).repeat(1, 1, hidden_size)
                            cur_cls_avg_embedding = (cur_cls_words_embeddings * random_tensor).sum(1,
                                                                                                   keepdim=True) / random_tensor.sum(
                                1, keepdim=True)
                        else:
                            cur_cls_avg_embedding = cur_cls_words_embeddings.mean(1, keepdim=True)
                else:
                    cur_cls_words_embeddings = cur_cls_words_embeddings.unsqueeze(0).repeat(num_seq, 1, 1)
                    # print('cur_cls_words_embeddings shape2: {}'.format(cur_cls_words_embeddings.shape))
                    if cls_id == len(probing_words_list) - 1:
                        # sep token
                        cur_cls_avg_embedding = cur_cls_words_embeddings.mean(1, keepdim=True)
                    else:
                        if global_file.args.dropout_prob > 0:
                            _, num_prob, hidden_size = cur_cls_words_embeddings.shape
                            random_tensor = torch.tensor(np.random.binomial(n=1, p=global_file.args.dropout_prob,
                                                                            size=(num_seq, num_prob))).cuda()
                            random_tensor = random_tensor.unsqueeze(-1).repeat(1, 1, hidden_size)
                            cur_cls_avg_embedding = (cur_cls_words_embeddings * random_tensor).sum(1,
                                                                                                   keepdim=True) / random_tensor.sum(
                                1, keepdim=True)
                        else:
                            cur_cls_avg_embedding = cur_cls_words_embeddings.mean(1, keepdim=True)
                        # print('cur_cls_avg_embedding shape: {}'.format(cur_cls_avg_embedding.shape))
                        # print('words_embeddings shape: {}'.format(words_embeddings.shape))
                # words_embeddings = torch.cat((words_embeddings, cur_cls_avg_embedding), 1)
                probing_words_embedding.append(cur_cls_avg_embedding.squeeze())
            # seq_length += len(probing_words_list)
            # print('word embeddings shape:{}'.format(words_embeddings.shape))

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(num_seq, seq_length)
        if token_type_ids is None:
            token_type_ids = torch.zeros((num_seq, seq_length), dtype=input_ids.dtype, device=input_ids.device)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        if probing_words_list is not None:
            return embeddings, probing_words_embedding
        else:
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

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                probing_words_list=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        # if token_type_ids is None:
        #     token_type_ids = torch.zeros_like(input_ids)

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
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
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
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers
        if probing_words_list is None:
            embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                               probing_words_list=probing_words_list)
        else:
            embedding_output, probing_words_embedding = self.embeddings(input_ids, position_ids=position_ids,
                                                                        token_type_ids=token_type_ids,
                                                                        probing_words_list=probing_words_list)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
                                                      1:]  # add hidden_states and attentions if they are here
        if probing_words_list is None:
            return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)
        else:
            return outputs, probing_words_embedding


class ClassificationBertWithProbingWord(nn.Module):
    def __init__(self, num_labels, model='bert-base-uncased'):
        super().__init__()
        config = BertConfig.from_pretrained(
            './cache/pytorch_transformers/4dad0251492946e18ac39290fcfe91b89d370fee250efe9521476438fe8ca185.7156163d5fdc189c3016baca0775ffce230789d7fa2a42ef516483e4ca884517',
            output_hidden_states=True, output_attentions=True,
            cache_dir=None, force_download=False)
        self.bert = BertModelAvgPrb.from_pretrained(
            './cache/pytorch_transformers/aa1ef1aede4482d0dbcd4d52baad8ae300e60902e88fcb0bebdec09afd232066.36ca03ab34a1a5d5fa7bc3d03d55c4fa650fed07220e2eeebc06ce58d0e9a157',
            config=config)
        self.classifier = nn.Sequential(nn.Linear(self.bert.config.hidden_size, 128),
                                        nn.Tanh(),
                                        nn.Linear(128, num_labels))
        # self.prob_head_name = 'prob_head'
        # for cls_id in range(num_labels):
        #     cur_block_name = self.prob_head_name + str(cls_id)
        #     setattr(self, cur_block_name, nn.Sequential(nn.Linear(self.bert.config.hidden_size, 128),
        #                                                 nn.Tanh(),
        #                                                 nn.Linear(128, 1)))
        self.prob_match_cls = nn.Sequential(nn.Linear(self.bert.config.hidden_size * 2, 128),
                                            nn.Tanh(),
                                            nn.Linear(128, 1))
        self.num_labels = num_labels

    def forward(self, token_id, text_len, probing_words_list=None, use_cls_sep=False, output_attention=False):
        num_seq, seq_length = token_id.size()
        # if split_match:
        #     added_len = 2 if use_cls_sep else 1
        #     len_range = torch.arange(seq_length + added_len, device=token_id.device,
        #                              dtype=text_len.dtype).expand(num_seq, seq_length + added_len)
        #     mask1 = (len_range < text_len.unsqueeze(-1)).long()
        #     mask1[:, -added_len:] += 1
        #     seq_logits_list, prob_head_logits = list(), list()
        #     for cls_id in range(self.num_labels):
        #         cur_probing_words = list()
        #         cur_probing_words.append(probing_words_list[cls_id])
        #         if use_cls_sep:
        #             cur_probing_words.append(probing_words_list[-1])
        #         out = self.bert(token_id, attention_mask=mask1, probing_words_list=cur_probing_words)[0]
        #
        #         if use_cls_sep:
        #             mask1[:, -added_len - 1:] -= 1
        #             mask1[:, 0] -= 1
        #         else:
        #             mask1[:, -added_len] -= 1
        #         seq_fea = out * mask1.unsqueeze(-1).expand(num_seq, seq_length + added_len,
        #                                                    self.bert.config.hidden_size)
        #         seq_gap_fea = torch.sum(seq_fea, 1) / mask1.sum(1, keepdim=True)
        #         seq_logits_list.append(self.classifier(seq_gap_fea))
        #
        #         cur_match_fea = torch.cat((seq_gap_fea, out[:, -added_len]), 1)
        #         prob_head_logits.append(self.prob_match_cls(cur_match_fea))
        #     prob_head_logits = torch.stack(prob_head_logits, 1).squeeze()
        #     return seq_logits_list, prob_head_logits
        len_range = torch.arange(seq_length, device=token_id.device, dtype=text_len.dtype).expand(num_seq,
                                                                                                  seq_length)
        mask1 = (len_range < text_len.unsqueeze(-1)).long()
        if probing_words_list is None:
            out = self.bert(token_id, attention_mask=mask1, probing_words_list=probing_words_list)

            if use_cls_sep:
                # ======== gap on feature only (assume cls&sep are used) ==========
                mask = (len_range < (text_len - 1).unsqueeze(-1)).long()
                mask[:, 0] = 0
                masked_out = out[0] * mask.unsqueeze(-1).expand(*out[0].size())
                seq_gap_fea = torch.sum(masked_out, 1) / (text_len - 2).unsqueeze(-1)
            else:
                # ======== gap on all output sequence ========
                seq_gap_fea = torch.mean(out[0], 1)
            seq_logits = self.classifier(seq_gap_fea)
            if output_attention:
                return seq_logits, out[-1]
            else:
                return seq_logits
        else:
            out, probing_words_embedding = self.bert(token_id, attention_mask=mask1,
                                                     probing_words_list=probing_words_list)
            if use_cls_sep:
                # ======== gap on feature only (assume cls&sep are used) ==========
                mask = (len_range < (text_len - 1).unsqueeze(-1)).long()
                mask[:, 0] = 0
                masked_out = out[0] * mask.unsqueeze(-1).expand(*out[0].size())
                seq_gap_fea = torch.sum(masked_out, 1) / (text_len - 2).unsqueeze(-1)
            else:
                # ======== gap on all output sequence ========
                seq_gap_fea = torch.mean(out[0], 1)
            seq_logits = self.classifier(seq_gap_fea)

            prob_head_logits = list()
            for cls_id in range(self.num_labels):
                # if global_file.args.multiple_sep:
                #     cur_match_fea = torch.cat((seq_gap_fea, probing_words_embedding[cls_id * 2].detach()), 1)
                # else:
                #     cur_match_fea = torch.cat((seq_gap_fea, probing_words_embedding[cls_id].detach()), 1)
                # prob_head_logits.append(self.prob_match_cls(cur_match_fea))

                cur_match_logit = (seq_gap_fea * probing_words_embedding[cls_id]).sum(1)
                prob_head_logits.append(cur_match_logit)

            prob_head_logits = torch.stack(prob_head_logits, 1).squeeze()
            if len(prob_head_logits.size()) == 1:
                prob_head_logits = prob_head_logits.unsqueeze(0)
            if output_attention:
                return seq_logits, prob_head_logits, out[-1]
            else:
                return seq_logits, prob_head_logits
