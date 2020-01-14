import torch
import torch.nn as nn

from model.ops import gelu, build_attn_mask
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
        self.lm = nn.Linear(params.hidden_dim, params.vocab_size, bias=False)
        self.lm.weight = self.embedding.tok_embedding.weight  # Weight sharing
        self.lm_bias = nn.Parameter(torch.rand(params.vocab_size))

    def forward(self, input_ids, segment_ids, masked_pos):
        # input_ids    = [batch size, sentence length]
        # segment_ids  = [batch size, sentence length]
        # masked_pos   = [batch size, max pred]
        
        output = self.embedding(input_ids, segment_ids)
        # output       = [batch size, sentence length, hidden dim]

        attn_mask = build_attn_mask(input_ids)
        for layer in self.layers:
            output = layer(output, attn_mask)
        # output       = [batch size, sentence length, hidden dim]

        cls_tokens = self.activ1(self.fc(output[:, 0]))
        # cls_tokens   = [batch size, hidden dim]
        cls_logits = self.classifier(cls_tokens)
        # cls_logits   = [batch size, 2]

        # Expand maksed pos hidden dimenstion times to match dimension
        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1))
        # masked_pos   = [batch size, max pred, hidden dim]

        # Extract masked tokens from final output sentence representation
        masked_tokens = torch.gather(output, 1, masked_pos)
        masked_tokens = self.norm(self.activ2(self.linear(masked_tokens)))
        # masked_tokens = [batch size, max pred, hidden dim]
        
        lm_logits = self.lm(masked_tokens) + self.lm_bias
        # lm_logits     = [batch size, max pred, vocab size]

        return lm_logits, cls_logits
        
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)