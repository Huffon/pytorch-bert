import json
import argparse

import torch

from trainer import Trainer
from utils import build_iter


def main(params):
    if params.mode == 'train':
        train_iter, valid_iter = build_iter(params, 'train')
        trainer = Trainer(params, train_iter=train_iter, valid_iter=valid_iter)
        trainer.train()
    else:
        test_iter = build_iter(params, 'test')
        trainer = Trainer(params, test_iter=test_iter)
        trainer.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BERT')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--num_epoch', type=int, default=2)
    parser.add_argument('--bsz', type=int, default=32)

    # Hyper-parameters
    parser.add_argument('--max_len', type=int, default=32)
    parser.add_argument('--max_pred', type=int, default=5)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--num_segments', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--ffn_dim', type=int, default=3072)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Add pre-built vocab size to params
    vocab = json.load(open('vocab.json'))
    parser.add_argument('--vocab_size', type=int, default=len(vocab))
    
    # Add device information
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--device', type=str, default=device)

    args = parser.parse_args()
    main(args)
