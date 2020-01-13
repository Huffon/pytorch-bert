import time
import random

import torch
import torch.nn as nn
import torch.optim as optim

from model.net import BERT

random.seed(32)
torch.manual_seed(32)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:
    def __init__(self, params, train_iter=None, valid_iter=None, test_iter=None):
        self.params = params

        if params.mode == 'train':
            self.train_iter = train_iter
            self.valid_iter = valid_iter
        
        else:
            self.test_iter = test_iter
        
        self.model = BERT(self.params)
        self.model.to(device)

        self.optimizer = optim.Adam(self.model.parameters())

        self.criterion = nn.CrossEntropyLoss()
        self.criterion.to(device)

    def train(self):
        print(self.model)
        print(f'The model has {self.model.count_params():,} trainable parameters')
        best_valid_loss = float('inf')
        
        for epoch in range(self.params.num_epoch):
            self.model.train()
            epoch_loss = 0
            start_time = time.time()
            
            for batch in self.train_iter:
                self.optimizer.zero_grad()
                lm_logits, cls_logits = self.model(x, segment, masked_pos)

                loss_lm = self.criterion()
                loss_lm = (loss_lm.float()).mean()

                loss_cls = self.criterion(cls_logits, isNext)

                loss = loss_lm + loss_cls
                loss.backward()
                self.optimizer.step()

    def valid(self):
        pass

    def test(self):
        pass