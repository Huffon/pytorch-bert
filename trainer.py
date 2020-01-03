import random

import torch
import torch.nn as nn
import torch.optim as optim

from model.net import BERT

random.seed(32)
torch.manual_seed(32)
torch.backends.cudnn.deterministic = True


class Trainer:
    def __init__(self, params, mode, train_iter=None, valid_iter=None, test_iter=None):
        self.params = params

        if mode == 'train':
            self.train_iter = train_iter
            self.valid_iter = valid_iter
        
        else:
            self.test_iter = test_iter
        
        self.model = BERT(self.params)
        self.model.to(self.params.device)

        self.opt = optim.Adam(self.model.parameters())

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.param.pad_idx)
        self.criterion.to(self.params.device)

    def train(self):
        pass

    def valid(self):
        pass

    def test(self):
        pass