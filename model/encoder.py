import torch.nn as nn

from model.sub_layers import MultiHeadAttention
from model.sub_layers import PositionWiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, params):
        supre(EncoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6)
        self.self_attention = MultiHeadAttention(params)
        self.position_wise_ffn = PositionWiseFeedForward(params)

    def forward(self, x):
        # x = [batch size, sentence length, hidden dim]

        output = x + self.self_attention(x, x, x)
        output = output + self.position_wise_ffn(output)
        
        return output


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList([EncoderLayer(params) for _ in range(params.num_layer)])
        self.dropout = nn.Dropout(params.dropout)
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6)

    def forward(self, x):
        # x = [batch size, sentence length]

        x = self.token_embedding(x)
        x = self.dropout(x + self.pos_embedding(x))
        # x = [batch size, sentence length, hidden dim]

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        return self.layer_norm(x)
