import torch.nn as nn

from sub_layers import Embedding
from encoder import Encoder


class BERT(nn.Module):
    def __init__(self, params):
        super(BERT, self).__init__()
        self.embedding = Embedding()
        self.encoder = Encoder(params)



    def forward(self, x, segments):
        # x = [batch size, sentence length]

        embedded = self.embedding(x, segments)
        # embedded = [batch size, sentence length, hidden dim]

        output = self.encoder(output)
        # output = [batch size, sentene length, hidden dim]

        return 
        