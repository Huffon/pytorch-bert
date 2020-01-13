import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ops import gelu, SelfAttention


class Embedding(nn.Module):
    def __init__(self, params):
        super(Embedding, self).__init__()
        self.tok_embedding = nn.Embedding(params.vocab_size, params.hidden_dim)
        self.pos_embedding = nn.Embedding(params.max_len, params.hidden_dim)
        self.seg_embedding = nn.Embedding(params.num_segments, params.hidden_dim)
        self.layer_norm = nn.LayerNorm(params.hidden_dim)

    def forward(self, x, seg):
        # x = [batch size, sentence length]
        # seg = [batch size, sentence length]
        bsz, sent_len = x.size()
        
        position = torch.arange(sent_len, dtype=torch.long)
        # position = [sentence length]: (0 ~ sentence length-1)
        
        position = position.repeat(bsz, 1)
        # position = [batch size, sentence length]

        embed = self.tok_embedding(x) + self.pos_embedding(position) + self.seg_embedding(seg)
        return self.layer_norm(embed)


class EncoderLayer(nn.Module):
    def __init__(self, params):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(params)
        self.position_wise_ffn = PositionWiseFeedForward(params)

    def forward(self, x, attn_mask):
        # x = [batch size, sentence length, hidden dim]
        output = self.self_attention(x, x, x, attn_mask)
        output = self.position_wise_ffn(output)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, params):
        super(MultiHeadAttention, self).__init__()
        assert params.hidden_dim % params.num_heads == 0
        self.attentions = nn.ModuleList([SelfAttention(params) for _ in range(params.num_heads)])
        self.o_w = nn.Linear(params.hidden_dim, params.hidden_dim, bias=False)
        self.dropout = nn.Dropout(params.dropout)

    def forward(self, query, key, value, mask=None):
        # query, key, value = [batch size, sentence length, hidden dim]

        weighted_v = [attention(query, key, value, mask) for attention in self.attentions]
        # weighted_v = [batch size, sentence length, attention dim] * num head
        weighted_v = torch.cat(weighted_v, dim=-1)
        # weighted_v = [batch size, sentence length, hidden dim]
        output = self.dropout(self.o_w(weighted_v))
        # output = [batch size, sentence length, hidden dim]
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, params):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(params.hidden_dim, params.ffn_dim)
        self.fc2 = nn.Linear(params.ffn_dim, params.hidden_dim)
        self.dropout = nn.Dropout(params.dropout)
    
    def forward(self, x):
        # x = [batch size, sentence length, hidden dim]
        output = self.dropout(gelu(self.ff1(x)))  # [batch size, sentence length, feed forward dim]
        output = self.ff2(output)                 # [batch size, sentence length, hidden dim]
        return self.dropout(output)