import math

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, params):
        super(SelfAttention, self).__init__()
        self.hidden_dim = params.hidden_dim
        self.attention_dim = params.hidden_dim // params.n_head

        self.q_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        self.k_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        self.v_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)

        self.dropout = nn.Dropout(params.dropout)
        # self.scale_factor = 

    def forward(self, query, key, value, mask=None):
        # query, key, value = [batch size, sentence length, hidden dim]

        q = self.q_w(query)
        k = self.k_w(key)
        v = self.v_w(value)
        # q, k, v = [batch size, sentence length, attention dim]

        self_attention = torch.bmm(q, k.permute(0, 2, 1))
        self_attention = self_attention / self.scale_factor
        # self_attention = [batch size, sentence length, sentence length]

        # if mask is not None:

        attention_score = F.softmax(self_attention, dim=-1)
        attention_score = self.dropout(attention_score)

        weighted_v = torch.bmm(attention_score, v)
        # weighted_v = [batch size, sentence length, attention dim]

        return self.dropout(weighted_v)


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
