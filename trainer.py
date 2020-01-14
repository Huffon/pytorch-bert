import time
import json
import random

import torch
import torch.nn as nn
import torch.optim as optim

from model.net import BERT

random.seed(32)
torch.manual_seed(32)
torch.backends.cudnn.deterministic = True


class Trainer:
    def __init__(self, params, train_iter=None, valid_iter=None, test_iter=None):
        self.params = params

        if params.mode == 'train':
            self.train_iter = train_iter
            self.valid_iter = valid_iter
        
        else:
            self.test_iter = test_iter
        
        self.model = BERT(self.params)
        self.model.to(params.device)

        self.optimizer = optim.Adam(self.model.parameters())

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.criterion.to(params.device)

    def train(self):
        # print(self.model)
        # print(f'The model has {self.model.count_params():,} trainable parameters')
        best_valid_loss = float('inf')

        # For presentation
        f_vocab = open('vocab.json')
        vocab = json.load(f_vocab)
        idx_vocab = {i: w for i, w in enumerate(vocab)}
        
        for epoch in range(self.params.num_epoch):
            self.model.train()
            epoch_loss = 0
            start_time = time.time()
            
            # for batch in self.train_iter: # batchify
            input_ids, segment_ids, masked_tokens, masked_pos, isNext = self.train_iter
            self.optimizer.zero_grad()
            lm_logits, cls_logits = self.model(input_ids, segment_ids, masked_pos)
            
            # Calculate losses for Masked LM
            loss_lm = self.criterion(lm_logits.transpose(1, 2), masked_tokens)
            # lm_logits     = [batch size, vocab size, max pred]
            # masked_tokens = [batch size, max pred]
            loss_lm = (loss_lm.float()).mean()

            # Calculate losses for Next Sentence Prediction
            loss_cls = self.criterion(cls_logits, isNext)
            loss = loss_lm + loss_cls

            # For presentation
            rand_idx = random.randrange(len(input_ids))
            first_sent = [idx_vocab[idx] for idx in input_ids[rand_idx].cpu().numpy()]
            mask_toks = [idx_vocab[idx] for idx in masked_tokens[rand_idx].cpu().numpy()]
            pred_toks = [idx_vocab[idx] for idx in torch.argmax(lm_logits[rand_idx], 1).cpu().numpy()]
            print(f'Masked sentence: \n{" ".join(first_sent)}')
            print(f'Masked tokens: {mask_toks}')
            print(f'Masked positions: {masked_pos[rand_idx].cpu().numpy()}')
            print(f'Is next?: {bool(isNext[rand_idx])}')
            print(f'---------------------------------------')
            print(f'[Predict] Masked tokens: {pred_toks}')
            print(f'[Predict] Is next: {torch.argmax(cls_logits[rand_idx])}')
            print(f'---------------------------------------')
            loss.backward()
            self.optimizer.step()

    def valid(self):
        self.model.eval()
        epoch_loss = 0

        with torch.no_grad():
            pass

    def test(self):
        self.model.eval()
        epoch_loss = 0

        with torch.no_grad():
            pass