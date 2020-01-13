import argparse

from trainer import Trainer
from utils import load_data


def main(params):
    if params.mode == 'train':
        train_iter, valid_iter = load_data('train')
        trainer = Trainer(params, train_iter=train_iter, valid_iter=valid_iter)
        trainer.train()
    else:
        test_iter = load_data('test')
        trainer = Trainer(params, test_iter=test_iter)
        trainer.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BERT')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--num_epoch', type=int, default=3)
    parser.add_argument('--bsz', type=int, default=32)
    # Hyper-parameters
    parser.add_argument('--max_len', type=int, default=32)
    parser.add_argument('--max_pred', type=int, default=5)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--num_segments', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--ffn_dim', type=int, default=3072)
    
    args = parser.parse_args()
    main(args)
