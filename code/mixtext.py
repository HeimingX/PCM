import torch
import torch.nn as nn
from pytorch_transformers import *
from transformers.modeling_bert import BertEmbeddings, BertPooler, BertLayer, BertLMPredictionHead
# from transformers import BertConfig
import ipdb


class BertModel4Mix(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel4Mix, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder4Mix(config)
        self.pooler = BertPooler(config)

        self.predictions = BertLMPredictionHead(config)
        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(
            old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, input_ids2=None, l=None, mix_layer=1000, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, mlm_output=False):

        if attention_mask is None:
            if input_ids2 is not None:
                attention_mask2 = torch.ones_like(input_ids2)
            attention_mask = torch.ones_like(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
            if input_ids2 is not None:
                token_type_ids2 = torch.zeros_like(input_ids2)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if input_ids2 is not None:
            extended_attention_mask2 = attention_mask2.unsqueeze(1).unsqueeze(2)
            extended_attention_mask2 = extended_attention_mask2.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask2 = (1.0 - extended_attention_mask2) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                # We can specify head_mask for each layer
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            # switch to fload if need + fp16 compatibility
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        if input_ids2 is not None:
            embedding_output2 = self.embeddings(input_ids2, position_ids=position_ids, token_type_ids=token_type_ids2)

        if input_ids2 is not None:
            encoder_outputs = self.encoder(embedding_output, embedding_output2, l, mix_layer,
                                           extended_attention_mask, extended_attention_mask2, head_mask=head_mask)
        else:
            encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask)

        sequence_output = encoder_outputs[0]

        if mlm_output:
            return self.predictions(sequence_output)
        else:
            pooled_output = self.pooler(sequence_output)

            # add hidden_states and attentions if they are here
            outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
            # sequence_output, pooled_output, (hidden_states), (attentions)
            return outputs


class BertEncoder4Mix(nn.Module):
    def __init__(self, config):
        super(BertEncoder4Mix, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, hidden_states2=None, l=None, mix_layer=1000, attention_mask=None,
                attention_mask2=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()

        # Perform mix at till the mix_layer
        if mix_layer == -1:
            if hidden_states2 is not None:
                hidden_states = l * hidden_states + (1 - l) * hidden_states2

        for i, layer_module in enumerate(self.layer):
            if i <= mix_layer:

                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i])
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

                if hidden_states2 is not None:
                    layer_outputs2 = layer_module(
                        hidden_states2, attention_mask2, head_mask[i])
                    hidden_states2 = layer_outputs2[0]

            if i == mix_layer:
                if hidden_states2 is not None:
                    hidden_states = l * hidden_states + (1 - l) * hidden_states2

            if i > mix_layer:
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i])
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        # last-layer hidden state, (all hidden states), (all attentions)
        return outputs


class MixText(nn.Module):
    def __init__(self, num_labels=2, mix_option=False, require_attention=False):
        super(MixText, self).__init__()
        self.require_attention = require_attention
        config = BertConfig.from_pretrained(
                './cache/pytorch_transformers/4dad0251492946e18ac39290fcfe91b89d370fee250efe9521476438fe8ca185.7156163d5fdc189c3016baca0775ffce230789d7fa2a42ef516483e4ca884517',
                cache_dir=None, force_download=False, output_hidden_states=require_attention,
                output_attentions=require_attention)
        assert mix_option, 'mix_option should be True!'
        if mix_option:
            self.bert = BertModel4Mix.from_pretrained('./cache/pytorch_transformers/aa1ef1aede4482d0dbcd4d52baad8ae300e60902e88fcb0bebdec09afd232066.36ca03ab34a1a5d5fa7bc3d03d55c4fa650fed07220e2eeebc06ce58d0e9a157', config=config)
        else:
            self.bert = BertModel.from_pretrained('./cache/pytorch_transformers/aa1ef1aede4482d0dbcd4d52baad8ae300e60902e88fcb0bebdec09afd232066.36ca03ab34a1a5d5fa7bc3d03d55c4fa650fed07220e2eeebc06ce58d0e9a157', config=config)

        self.linear = nn.Sequential(nn.Linear(768, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, num_labels))

        self.gradients = None
        self.activations = None
        self.attentions = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x, x2=None, l=None, mix_layer=1000, register_hook=False, mlm=False, output_attention=False, attention_mask=None):
        # ipdb.set_trace()
        if x2 is not None:
            if self.require_attention:
                all_hidden, pooler, _, _ = self.bert(x, x2, l, mix_layer)
            else:
                all_hidden, pooler = self.bert(x, x2, l, mix_layer)
            pooled_output = torch.mean(all_hidden, 1)
        else:
            if mlm:
                predict = self.bert(x, mlm_output=True, attention_mask=attention_mask)
                return predict
            else:
                if self.require_attention:
                    if output_attention:
                        all_hidden, pooler, _, attentions = self.bert(x)
                        self.attentions = attentions
                    else:
                        all_hidden, pooler, _, _ = self.bert(x)
                else:
                    all_hidden, pooler = self.bert(x)

                pooled_output = torch.mean(all_hidden, 1)

        if register_hook:
            self.activations = all_hidden
            h = all_hidden.register_hook(self.activations_hook)

        predict = self.linear(pooled_output)

        if output_attention:
            return predict, self.attentions
        else:
            return predict

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self):
        return self.activations

    def get_attentions(self):
        return self.attentions


