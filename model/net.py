import torch
import torch.nn as nn

from model.ops import gelu, get_attn_mask
from model.sublayers import Embedding, EncoderLayer


class BERT(nn.Module):
    def __init__(self, params):
        super(BERT, self).__init__()
        self.embedding = Embedding(params)
        self.layers = nn.ModuleList([EncoderLayer(params) for _ in range(params.num_layers)])
        # Logits classification
        self.fc = nn.Linear(params.hidden_dim, params.hidden_dim)
        self.activ1 = nn.Tanh()
        self.classifier = nn.Linear(params.hidden_dim, 2)
        # Masked Language Modeling        
        self.linear = nn.Linear(params.hidden_dim, params.hidden_dim)
        self.activ2 = gelu
        self.norm = nn.LayerNorm(params.hidden_dim)
        # Masked token classifier
        self.lm = nn.Linear(params.vocab_size, params.hidden_dim, bias=False)
        self.lm.weight = self.embedding.tok_embedding.weight  # Weight sharing
        self.lm_bias = nn.Parameter(torch.rand(params.vocab_size))

    def forward(self, x, segments, masked_pos):
        # x          = [batch size, sentence length]
        # segments   = [batch size, sentence length]
        # masked_pos = [batch size, max pred]
        embedded = self.embedding(x, segments)
        # embedded   = [batch size, sentence length, hidden dim]

        attn_mask = get_attn_mask(x, segments)
        for layer in self.layers:
            output = layer(output, attn_mask)
        # output     = [batch size, sentence length, hidden dim]

        cls_toks = self.activ1(self.fc(output[:, 0]))
        # cls_toks   = [batch size, hidden dim]
        cls_logits = self.classifier(cls_toks)
        # cls_logits = [batch size, 2]

        masked_pos = masked_pos
        masked_pos = masked_pos
        lm_logits = self.lm(h) + self.lm_bias

        return lm_logits, cls_logits
        
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)