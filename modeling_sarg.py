import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from transformers import PretrainedConfig, PreTrainedModel
from transformer_utils import Embedding, TransformerBlock


class SARGConfig(PretrainedConfig):
    model_type = "SARG"

    def __init__(self,
                 tag_size=3,
                 vocab_size=21128,
                 n_positions=512,
                 ctx_len=256,
                 src_len=256,
                 n_embd=768,
                 n_intermediate_size=3072,
                 n_layer=12,
                 n_head=12,
                 n_utterance=10,
                 activation_function="gelu",
                 resid_pdrop=0.1,
                 embd_pdrop=0.1,
                 attn_pdrop=0.1,
                 layer_norm_epsilon=1e-5,
                 initializer_range=0.02,
                 bos_token_id=101,
                 eos_token_id=102,
                 pad_token_id=0,
                 cov_weight=0.,
                 blend_gen=True,
                 blend_copy=True,
                 mix_neighbors=False,
                 entity_size=50,
                 add_entity=False,
                 change_weight=1.,
                 alpha=1.,
                 **kwargs):
        super(SARGConfig, self).__init__(bos_token_id=bos_token_id,
                                         eos_token_id=eos_token_id,
                                         pad_token_id=pad_token_id,
                                         **kwargs)
        self.tag_size = tag_size  # [KEEP], [DELETE], [CHANGE]
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.src_len = src_len
        self.ctx_len = ctx_len
        self.n_embd = n_embd
        self.n_intermediate_size = n_intermediate_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_utterance = n_utterance
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.cov_weight = cov_weight
        self.blend_gen = blend_gen
        self.blend_copy = blend_copy
        assert self.blend_copy or blend_gen
        self.mix_neighbors = mix_neighbors

        self.entity_size = entity_size
        self.add_entity = add_entity
        self.change_weight = change_weight
        self.alpha = alpha

    @property
    def max_position_embeddings(self):
        return self.n_positions

    @property
    def max_utterance_embeddings(self):
        return self.n_utterance

    @property
    def hidden_size(self):
        return self.n_embd

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_hidden_layers(self):
        return self.n_layer


class SARGPreTrainedModel(PreTrainedModel):
    config_class = SARGConfig
    base_model_prefix = "sarg"

    def __init__(self, *inputs, **kwargs):
        super(SARGPreTrainedModel, self).__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear,)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class CoverageAttention(nn.Module):
    def __init__(self, config: SARGConfig):
        super(CoverageAttention, self).__init__()
        self.linear_h = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_K = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_cov = nn.Linear(1, config.hidden_size)
        self.v = nn.Linear(config.hidden_size, 1)
        self.register_buffer("masked_bias", torch.tensor(-1e4))

    def forward(self, h, K, cov, mask=None):
        """
        :param K: (batch_size, src_len, hidden_size)
        :param h: (batch_size, hidden_size)
        :param cov: (batch_size, src_len)
        :param mask:
        :return:
        """
        h_l = self.linear_h(h.unsqueeze(1))  # (batch_size, 1, hidden_size)
        K_l = self.linear_K(K)
        c_l = self.linear_cov(cov.unsqueeze(2))
        e = self.v(torch.tanh(h_l + K_l + c_l)).squeeze(-1)  # (batch_size, src_len)

        if mask is not None:
            e = e + mask * self.masked_bias

        a = F.softmax(e, dim=-1).unsqueeze(1)  # (batch_size, 1, src_len)
        out = torch.matmul(a, K).squeeze(1)  # (batch_size, 1, hidden_size)
        a = a.squeeze(1)
        cov = cov + a
        return out, a, cov


class LSTMWithCovAttention(nn.Module):
    def __init__(self, config: SARGConfig):
        super(LSTMWithCovAttention, self).__init__()
        self.attn = CoverageAttention(config)
        self.lstm = nn.LSTMCell(input_size=config.hidden_size, hidden_size=config.hidden_size)
        self.dropout = nn.Dropout(config.resid_pdrop)
        self.config = config

        if config.blend_gen and config.blend_copy:
            self.vocab_linear = nn.Sequential(nn.Linear(2 * config.hidden_size, config.vocab_size),
                                              nn.Softmax(dim=-1))

            self.p_gen_wh = nn.Linear(config.hidden_size, 1)
            self.p_gen_ws = nn.Linear(config.hidden_size, 1)
            self.p_gen_wx = nn.Linear(config.hidden_size, 1)
        elif config.blend_gen:
            self.vocab_linear = nn.Sequential(nn.Linear(2 * config.hidden_size, config.vocab_size),
                                              nn.Softmax(dim=-1))

    def forward(self, x, states, K, cov, ctx_token, mask=None):
        """
        :param x:  (batch_size, hidden_size)
        :param states: h: (batch_size, hidden_size) c: (batch_size, hidden_size)
        :param K: (batch_size, K_len, hidden_size)
        :param cov: (batch_size, K_len)
        :param ctx_token: (batch_size, K_len)
        :param mask:
        :return:
        """
        batch_size = x.size(0)
        h, c = self.lstm(x, states)
        # cov attn
        out, a, cov = self.attn(h=h, K=K, cov=cov, mask=mask)

        out = self.dropout(out)

        # cov loss
        min_a_c, _ = torch.stack([a, cov], dim=1).min(dim=1)
        cov_loss = min_a_c.sum() / batch_size

        if self.config.blend_gen and self.config.blend_copy:
            p_vocab = self.vocab_linear(torch.cat([h, out], dim=-1))
            p_gen = torch.sigmoid(self.p_gen_wh(out) + self.p_gen_ws(h) + self.p_gen_wx(x))  # (batch_size, 1)
            p_vocab = p_vocab * p_gen
            a = (1. - p_gen) * a
            p_final = p_vocab.scatter_add(dim=-1, index=ctx_token, src=a)
        elif self.config.blend_copy:
            p_vocab = torch.zeros(batch_size, self.config.vocab_size, device=out.device, dtype=torch.float) + 1e-10
            p_final = p_vocab.scatter_add(dim=-1, index=ctx_token, src=a)
        else:
            p_final = self.vocab_linear(torch.cat([h, out], dim=-1))
        return p_final, (h, c), cov, cov_loss


class SARGModel(SARGPreTrainedModel):
    def __init__(self, config: SARGConfig):
        super(SARGModel, self).__init__(config)
        self.config = config
        self.embedding = Embedding(config)
        self.embed_layer_norm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.decoder = LSTMWithCovAttention(config)
        self.dropout = nn.Dropout(config.resid_pdrop)

        self.tag_linear = nn.Sequential(nn.Dropout(config.attn_pdrop),
                                        nn.Linear(config.hidden_size, config.tag_size))

        if self.config.mix_neighbors:
            self.mix_layer_norm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()

    def _create_mask(self, input_mask: torch.Tensor, auto_regressive=False):
        """
        :param input_mask: 2D tensor
        :param auto_regressive:
        :return: 3D mask
        """
        assert len(input_mask.size()) == 2
        seq_len = input_mask.size(1)
        input_mask = input_mask.unsqueeze(dim=-1)
        mask1 = input_mask.repeat(1, 1, seq_len)
        mask2 = mask1.transpose(1, 2)
        mask = mask1 * mask2

        if auto_regressive:
            seq_mask = self.sequence_mask[:seq_len, :seq_len]
            mask = mask * seq_mask
        mask = 1 - mask
        return mask

    def _join_mask(self, mask1: torch.Tensor, mask2: torch.Tensor):
        """
        Merge source attention mask and target attention mask.
        :param mask1: 3D
        :param mask2: 3D
        :return: 3D mask
        """
        batch_size = mask1.size(0)
        seq_len1 = mask1.size(1)
        seq_len2 = mask2.size(2)
        seq_len = seq_len1 + seq_len2

        mask_lu = mask1
        mask_ru = torch.ones(batch_size, seq_len1, seq_len2, dtype=torch.long).to(mask_lu.device)
        mask3 = mask2[:, :, :1].repeat(1, 1, seq_len1)
        mask4 = mask1[:, :1].repeat(1, seq_len2, 1)
        mask_lb = mask3 + mask4 - mask3 * mask4
        mask_rb = mask2
        mask_u = torch.cat([mask_lu, mask_ru], dim=2)
        mask_b = torch.cat([mask_lb, mask_rb], dim=2)
        mask = torch.cat([mask_u, mask_b], dim=1)
        return mask

    def _forward(self, inputs):
        ctx_token = inputs["ctx_token"]
        ctx_mask = inputs["ctx_mask"]
        ctx_pos = inputs["ctx_pos"]
        ctx_ute = inputs["ctx_ute"]

        src_token = inputs["src_token"]
        src_mask = inputs["src_mask"]
        src_pos = inputs["src_pos"]
        src_ute = inputs["src_ute"]

        ctx_ner = inputs["ctx_ner"]
        src_ner = inputs["src_ner"]

        target = inputs["target"]  # (batch_size, src_len, max_added_len)
        batch_size, src_len, max_added_len = target.size()
        ctx_len = ctx_token.size(1)

        tag_labels = target[..., 0]  # (batch_size, src_len) for sequence labeling

        # embed
        ctx_embed = self.embedding(ctx_token, ctx_pos, ctx_ute, ctx_ner)
        src_embed = self.embedding(src_token, src_pos, src_ute, src_ner)

        embed = torch.cat([ctx_embed, src_embed], dim=1)
        embed = self.embed_layer_norm(embed)

        # unified transformers (sequence labeling)
        ctx_mask = self._create_mask(ctx_mask)
        src_mask = self._create_mask(src_mask)
        tm_mask = self._join_mask(ctx_mask, src_mask)
        for layer in self.layers:
            embed = layer(embed, attention_mask=tm_mask)  # (batch_size, ctx_len + src_len, hidden_size)

        ctx_encode = embed[:, :ctx_len]   # (batch_size, ctx_len, hidden_size)
        src_encode = embed[:, -src_len:]  # (batch_size, src_len, hidden_size)
        tag_logits = self.tag_linear(src_encode)  # (batch_size, src_len, tag_size)

        tag_weight = torch.ones(self.config.tag_size, dtype=torch.float, device=src_token.device)
        tag_weight[-1] = self.config.change_weight

        tag_loss = F.cross_entropy(tag_logits.view(-1, self.config.tag_size), tag_labels.view(-1),
                                   weight=tag_weight)

        # generation
        if target.size(-1) == 1:
            gen_loss = torch.tensor(0)
            total_cov_loss = torch.tensor(0)
        elif target.size(-1) <= 3:
            raise ValueError("Check the preprocess, must be the form of '[CLS] ... [SEP]' ")
        else:

            if self.config.mix_neighbors:
                # mix neighbors
                src_encode_left = torch.cat(
                    [torch.zeros(batch_size, 1, self.config.hidden_size, device=src_encode.device, dtype=torch.float),
                     src_encode[:, :-1]],
                    dim=1)
                src_encode_right = torch.cat(
                    [src_encode[:, 1:],
                     torch.zeros(batch_size, 1, self.config.hidden_size, device=src_encode.device, dtype=torch.float)],
                    dim=1)
                src_encode = src_encode + src_encode_left + src_encode_right
                src_encode = self.mix_layer_norm(src_encode)

            batch_K = []
            batch_h = []
            batch_c = []
            batch_x_in = []
            batch_x_la = []
            batch_mask = []
            batch_ctx_token = []
            for bidx in range(batch_size):
                for sidx in range(src_len):
                    if target[bidx][sidx][0] != 2:
                        continue
                    else:
                        batch_K.append(ctx_encode[bidx].unsqueeze(0))
                        batch_h.append(src_encode[bidx][sidx].unsqueeze(0))
                        batch_c.append(src_encode[bidx][sidx].unsqueeze(0))
                        batch_x_in.append(target[bidx][sidx][1:-1].unsqueeze(0))
                        batch_x_la.append(target[bidx][sidx][2:].unsqueeze(0))
                        batch_mask.append(1. - inputs["ctx_mask"][bidx].unsqueeze(0))
                        batch_ctx_token.append(ctx_token[bidx].unsqueeze(0))

            batch_K = torch.cat(batch_K, dim=0)
            batch_h = torch.cat(batch_h, dim=0)
            batch_c = torch.cat(batch_c, dim=0)
            batch_x_in = torch.cat(batch_x_in, dim=0)
            batch_mask = torch.cat(batch_mask, dim=0)
            batch_ctx_token = torch.cat(batch_ctx_token, dim=0)
            batch_cov = torch.zeros(*batch_K.size()[:2], dtype=torch.float, device=batch_K.device)

            gen_labels = torch.cat(batch_x_la, dim=0)
            gen_probs = []
            total_cov_loss = 0.
            for xidx in range(batch_x_in.size(1)):
                batch_x = self.embedding.word_embeddings(batch_x_in[:, xidx])  # (batch_size, hidden_size)
                p_final, (batch_h, batch_c), batch_cov, cov_loss = \
                    self.decoder(x=batch_x, states=(batch_h, batch_c), K=batch_K, cov=batch_cov,
                                 ctx_token=batch_ctx_token, mask=batch_mask)
                gen_probs.append(p_final)
                total_cov_loss += cov_loss
            total_cov_loss /= batch_x_in.size(1)

            gen_probs = torch.stack(gen_probs, dim=1)

            gen_loss = F.nll_loss(gen_probs.log().view(-1, self.config.vocab_size),
                                  gen_labels.view(-1),
                                  ignore_index=self.config.pad_token_id)

        return self.config.alpha * tag_loss + gen_loss + self.config.cov_weight * total_cov_loss, tag_logits, \
               {"tag_loss": tag_loss, "gen_loss": gen_loss, "cov_loss": self.config.cov_weight * total_cov_loss}

    def forward(self, ctx_token, ctx_mask, ctx_pos, ctx_ute, src_token, src_mask, src_pos, src_ute, target, ctx_ner=None, src_ner=None):
        inputs = dict()
        inputs["ctx_token"] = ctx_token
        inputs["ctx_mask"] = ctx_mask
        inputs["ctx_pos"] = ctx_pos
        inputs["ctx_ute"] = ctx_ute

        inputs["src_token"] = src_token
        inputs["src_mask"] = src_mask
        inputs["src_pos"] = src_pos
        inputs["src_ute"] = src_ute

        inputs["target"] = target

        inputs["ctx_ner"] = ctx_ner if self.config.add_entity else None
        inputs["src_ner"] = src_ner if self.config.add_entity else None

        outputs = self._forward(inputs)
        return outputs

    def _init_state(self, inputs):
        state = {}
        output = {}
        ctx_token = inputs["ctx_token"]
        ctx_mask = inputs["ctx_mask"]
        ctx_pos = inputs["ctx_pos"]
        ctx_ute = inputs["ctx_ute"]

        src_token = inputs["src_token"]
        src_mask = inputs["src_mask"]
        src_pos = inputs["src_pos"]
        src_ute = inputs["src_ute"]

        if self.config.add_entity:
            ctx_ner = inputs["ctx_ner"]
            src_ner = inputs["src_ner"]
        else:
            ctx_ner = None
            src_ner = None

        batch_size, src_len = src_token.size()
        ctx_len = ctx_token.size(1)

        # embed
        ctx_embed = self.embedding(ctx_token, ctx_pos, ctx_ute, ctx_ner)
        src_embed = self.embedding(src_token, src_pos, src_ute, src_ner)

        embed = torch.cat([ctx_embed, src_embed], dim=1)
        embed = self.embed_layer_norm(embed)

        # unified transformers (sequence labeling)
        ctx_mask = self._create_mask(ctx_mask)
        src_mask = self._create_mask(src_mask)
        tm_mask = self._join_mask(ctx_mask, src_mask)
        for layer in self.layers:
            embed = layer(embed, attention_mask=tm_mask)  # (batch_size, ctx_len + src_len, hidden_size)

        ctx_encode = embed[:, :ctx_len]
        src_encode = embed[:, -src_len:]  # (batch_size, src_len, hidden_size)
        tag_logits = self.tag_linear(src_encode)  # (batch_size, src_len, tag_size)
        tag_pred = torch.argmax(tag_logits, dim=-1)
        output['tag'] = tag_pred  # (batch_size, src_len)

        if self.config.mix_neighbors:
            # mix neighbors
            src_encode_left = torch.cat(
                [torch.zeros(batch_size, 1, self.config.hidden_size, device=src_encode.device, dtype=torch.float),
                 src_encode[:, :-1]],
                dim=1)
            src_encode_right = torch.cat(
                [src_encode[:, 1:],
                 torch.zeros(batch_size, 1, self.config.hidden_size, device=src_encode.device, dtype=torch.float)],
                dim=1)
            src_encode = src_encode + src_encode_left + src_encode_right
            src_encode = self.mix_layer_norm(src_encode)

        bidx_sidx_to_idx = {}
        cnt = 0
        # prepare (K, ctx_token, ctx_mask, h, c)
        batch_K = []
        batch_ctx_token = []
        batch_h = []
        batch_c = []
        batch_ctx_mask = []
        for bidx in range(tag_pred.size(0)):
            for sidx in range(tag_pred.size(1)):
                if tag_pred[bidx][sidx] != 2:
                    continue
                else:
                    bidx_sidx_to_idx[(bidx, sidx)] = cnt
                    cnt += 1
                    batch_K.append(ctx_encode[bidx].unsqueeze(0))
                    batch_h.append(src_encode[bidx][sidx].unsqueeze(0))
                    batch_c.append(src_encode[bidx][sidx].unsqueeze(0))
                    batch_ctx_mask.append(1. - inputs["ctx_mask"][bidx].unsqueeze(0))
                    batch_ctx_token.append(ctx_token[bidx].unsqueeze(0))

        if len(batch_K) == 0:
            state["GEN"] = False
            return state, output

        state["GEN"] = True
        batch_K = torch.cat(batch_K, dim=0)
        batch_h = torch.cat(batch_h, dim=0)
        batch_c = torch.cat(batch_c, dim=0)
        batch_ctx_mask = torch.cat(batch_ctx_mask, dim=0)
        batch_ctx_token = torch.cat(batch_ctx_token, dim=0)
        batch_cov = torch.zeros(*batch_K.size()[:2], dtype=torch.float, device=batch_K.device)

        state["batch_size"] = batch_K.size(0)
        state["batch_K"] = batch_K
        state["batch_h"] = batch_h
        state["batch_c"] = batch_c
        state["batch_ctx_token"] = batch_ctx_token
        state["batch_ctx_mask"] = batch_ctx_mask
        state["batch_cov"] = batch_cov
        state["bidx_sidx_to_idx"] = bidx_sidx_to_idx

        return state, output

    def _decode(self, state):
        batch_x = state["batch_x"]  # (batch_size, 1)
        batch_h = state["batch_h"]
        batch_c = state["batch_c"]
        batch_ctx_token = state["batch_ctx_token"]
        batch_mask = state["batch_ctx_mask"]
        batch_K = state["batch_K"]
        batch_cov = state["batch_cov"]

        batch_x = self.embedding.word_embeddings(batch_x).squeeze(1)
        p_final, (batch_h, batch_c), batch_cov, cov_loss = \
            self.decoder(x=batch_x, states=(batch_h, batch_c), K=batch_K, cov=batch_cov,
                         ctx_token=batch_ctx_token, mask=batch_mask)

        state["batch_h"] = batch_h
        state["batch_c"] = batch_c
        state["batch_cov"] = batch_cov
        return p_final, state

    def _generate(self, step_fn, state, max_add_len, device, n_beams=1):
        pad_id = self.config.pad_token_id
        bos_id = self.config.bos_token_id
        batch_size = state["batch_size"]
        vocab_size = self.config.vocab_size
        unfinished_sents = torch.ones(batch_size, dtype=torch.long, device=device)
        sent_lengths = unfinished_sents.new(batch_size).fill_(max_add_len)

        predictions = bos_id * torch.ones(batch_size, 1, dtype=torch.long, device=device)

        for step in range(1, max_add_len + 1):
            pre_ids = predictions[:, -1:]
            state["batch_x"] = pre_ids
            scores, state = step_fn(state)
            preds = torch.argmax(scores, dim=-1).squeeze(-1)  # (batch_size, )
            if self.config.eos_token_id is not None:
                tokens_to_add = preds * unfinished_sents + self.config.pad_token_id * (1 - unfinished_sents)
            else:
                tokens_to_add = preds
            predictions = torch.cat([predictions, tokens_to_add.unsqueeze(dim=-1)], dim=-1)
            if self.config.eos_token_id is not None:
                eos_in_sents = tokens_to_add == self.config.eos_token_id
                # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, step)
                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())
            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break
        results = {
            "preds": predictions,
            "bidx_sidx_to_idx": state['bidx_sidx_to_idx']
        }
        return results

    def _generate_beam(self, step_fn, state, max_add_len, device, n_beams, length_average=True, length_penalty=0.0):
        def repeat(var, times):
            if isinstance(var, list):
                return [repeat(x, times) for x in var]
            elif isinstance(var, dict):
                return {k: repeat(v, times) for k, v in var.items()}
            elif isinstance(var, torch.Tensor):
                var = torch.unsqueeze(var, 1)
                expand_times = [1] * len(var.shape)
                expand_times[1] = times
                dtype = var.dtype
                var = var.to(torch.float).repeat(expand_times)
                shape = [var.size(0) * var.size(1)] + list(var.size())[2:]
                var = torch.reshape(var, shape).to(dtype)
                return var
            else:
                return var

        def gather(var, idx):
            if isinstance(var, list):
                return [gather(x, idx) for x in var]
            elif isinstance(var, dict):
                rlt = {}
                for k, v in var.items():
                    rlt[k] = gather(v, idx)
                return rlt
            elif isinstance(var, torch.Tensor):
                out = torch.index_select(var, dim=0, index=idx)
                return out
            else:
                return var

        pad_id = self.config.pad_token_id
        bos_id = self.config.bos_token_id
        eos_id = self.config.eos_token_id
        batch_size = state["batch_size"]
        vocab_size = self.config.vocab_size
        beam_size = n_beams

        pos_index = torch.arange(batch_size, dtype=torch.long, device=device) * beam_size
        pos_index = pos_index.unsqueeze(1)  # (batch_size, 1)

        predictions = torch.ones(batch_size, beam_size, 1, dtype=torch.long, device=device) * bos_id

        # initial input
        state["batch_x"] = predictions[:, 0]

        # (batch_size, vocab_size)
        scores, state = step_fn(state)

        eos_penalty = np.zeros(vocab_size, dtype="float32")
        eos_penalty[eos_id] = -1e10
        eos_penalty = torch.tensor(eos_penalty, device=device)

        scores_after_end = np.full(vocab_size, -1e10, dtype="float32")
        scores_after_end[pad_id] = 0
        scores_after_end = torch.tensor(scores_after_end, device=device)

        scores = scores + eos_penalty

        # preds: (batch_size, beam_size)
        # sequence_scores: (batch_size, beam_size)
        # initialize beams
        sequence_scores, preds = torch.topk(scores, beam_size)

        predictions = torch.cat([predictions, preds.unsqueeze(2)], dim=2)
        state = repeat(state, beam_size)

        for step in range(2, max_add_len + 1):
            pre_ids = predictions[:, :, -1:]
            state["batch_x"] = torch.reshape(pre_ids, shape=[batch_size * beam_size, 1])

            scores, state = step_fn(state)

            # Generate next
            # scores: (batch_size, beam_size, vocab_size)
            scores = torch.reshape(scores, shape=(batch_size, beam_size, vocab_size))

            # previous tokens is pad or eos
            pre_eos_mask = (pre_ids == eos_id).float().to(device) + (pre_ids == pad_id).float().to(device)
            if pre_eos_mask.sum() == beam_size * batch_size:
                # early stopping
                break
            scores = scores * (1 - pre_eos_mask) + pre_eos_mask.repeat(1, 1, vocab_size) * scores_after_end

            sequence_scores = sequence_scores.unsqueeze(2)

            if length_average:
                scaled_value = pre_eos_mask + (1 - pre_eos_mask) * (1 - 1 / step)
                sequence_scores = sequence_scores * scaled_value
                scaled_value = pre_eos_mask + (1 - pre_eos_mask) * (1 / step)
                scores = scores * scaled_value
            if length_penalty > 0.0:
                scaled_value = pre_eos_mask + (1 - pre_eos_mask) * \
                               (math.pow((4 + step) / (5 + step), length_penalty))
                sequence_scores = scaled_value * sequence_scores
                scaled_value = pre_eos_mask + (1 - pre_eos_mask) * \
                               (math.pow(1 / (5 + step), length_penalty))
                scores = scores * scaled_value

            # broadcast: every sequence combines with every potential word
            scores = scores + sequence_scores
            scores = scores.reshape(batch_size, beam_size * vocab_size)

            # update beams
            topk_scores, topk_indices = torch.topk(scores, beam_size)
            parent_idx = topk_indices // vocab_size  # (batch_size, beam_size)
            preds = topk_indices % vocab_size

            # gather state / sequence_scores
            parent_idx = parent_idx + pos_index
            parent_idx = parent_idx.view(-1)
            state = gather(state, parent_idx)
            sequence_scores = topk_scores

            predictions = predictions.reshape(batch_size * beam_size, step)
            predictions = gather(predictions, parent_idx)
            predictions = predictions.reshape(batch_size, beam_size, step)
            predictions = torch.cat([predictions, preds.unsqueeze(2)], dim=2)

        pre_ids = predictions[:, :, -1]
        pre_eos_mask = (pre_ids == eos_id).float().to(device) + (pre_ids == pad_id).float().to(device)
        sequence_scores = sequence_scores * pre_eos_mask + (1 - pre_eos_mask) * -1e10

        _, indices = torch.sort(sequence_scores, dim=1)
        indices = indices + pos_index
        indices = indices.view(-1)
        sequence_scores = torch.reshape(sequence_scores, [batch_size * beam_size])
        predictions = torch.reshape(predictions, [batch_size * beam_size, -1])
        sequence_scores = gather(sequence_scores, indices)
        predictions = torch.index_select(predictions, 0, indices)
        sequence_scores = torch.reshape(sequence_scores, [batch_size, beam_size])
        predictions = torch.reshape(predictions, [batch_size, beam_size, -1])

        results = {
            "preds": predictions[:, -1],
            "bidx_sidx_to_idx": state['bidx_sidx_to_idx']
        }
        return results

    def _infer(self, inputs, max_add_len, **kwargs):
        results = {}
        device = inputs["src_token"].device

        state, output = self._init_state(inputs)
        results.update(output)
        if state["GEN"] is False:
            return results

        if kwargs.get('n_beams', 1) == 1:
            gen_results = self._generate(self._decode, state, max_add_len, device)
        else:
            gen_results = self._generate_beam(self._decode, state, max_add_len, device, **kwargs)
        results.update(gen_results)
        return results

    def infer(self, inputs, max_add_len, **kwargs):
        self.eval()
        results = self._infer(inputs, max_add_len, **kwargs)
        results = {name: results[name].detach().cpu().numpy()
                   if isinstance(results[name], torch.Tensor) else results[name] for name in results}
        return results
