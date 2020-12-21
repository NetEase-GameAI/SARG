import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.pos_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.ute_embeddings = nn.Embedding(config.max_utterance_embeddings, config.hidden_size)
        if config.add_entity:
            self.ner_embeddings = nn.Embedding(config.entity_size, config.hidden_size)
        self.dropout = nn.Dropout(config.embd_pdrop)

    def forward(self, input_ids, pos_ids, ute_ids, ner_ids=None):
        embed = self.word_embeddings(input_ids) + \
                self.pos_embeddings(pos_ids) + \
                self.ute_embeddings(ute_ids)
        if self.config.add_entity:
            embed += self.ner_embeddings(ner_ids)
        embed = self.dropout(embed)
        return embed


class MultiHeadAttention(nn.Module):
    def __init__(self, config, scale=True):
        super(MultiHeadAttention, self).__init__()
        self.output_attentions = config.output_attentions

        nx = config.hidden_size

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.c_attn = nn.Linear(nx, n_state * 3)
        self.c_proj = nn.Linear(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)

    def _attn(self, q, k, v, attention_mask=None):
        w = torch.matmul(q, k)  # (batch, head, seq_length, seq_length)
        if len(attention_mask.size()) == 3:
            attention_mask = attention_mask.unsqueeze(dim=1).repeat(1, self.n_head, 1, 1).to(w.device)
            attention_mask = attention_mask * self.masked_bias

        if self.scale:
            w = w / (float(v.size(-1)) ** 0.5)

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        outputs = [torch.matmul(w, v)]  # (batch, head, seq_length, head_features)
        if self.output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, attention_mask=None, cache=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        if cache is not None:
            if "key" in cache and "value" in cache:
                key = torch.cat([cache["key"], key], dim=3)  # key is transposed
                value = torch.cat([cache["value"], value], dim=2)
            # cache本身是字典的键值，变化可以保持
            cache["key"] = key
            cache["value"] = value

        attn_outputs = self._attn(query, key, value, attention_mask)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)

        outputs = a
        return outputs


class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.hidden_size = config.hidden_size
        self.inner_size = config.n_intermediate_size

        self.zoomin = nn.Linear(self.hidden_size, self.inner_size)
        self.act = nn.GELU()
        self.zoomout = nn.Linear(self.inner_size, self.hidden_size)

        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        out = self.act(self.zoomin(x))
        out = self.dropout(out)
        out = self.zoomout(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super(TransformerBlock, self).__init__()
        nx = config.n_embd
        self.attn = MultiHeadAttention(config)
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.ff = FeedForward(config)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x, attention_mask=None, cache=None):
        attn_out = self.attn(x, attention_mask, cache)
        attn_out = self.dropout(attn_out)
        attn_out = self.ln_1(attn_out + x)

        ff_out = self.ff(attn_out)
        ff_out = self.dropout(ff_out)
        ff_out = self.ln_2(ff_out + attn_out)

        return ff_out
