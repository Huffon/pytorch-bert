import re
import json

import torch

torch.manual_seed(32)


def build_iter():
    pass


def build_vocab():
    """
    Build a vocabulary for designated corpus
    """
    corpus = open('data/train.txt', 'r', encoding='utf-8')
    lines = ''.join(corpus.readlines())

    lines = re.sub('[.,!?\\-]', '', lines.lower()).split('\n')  # filter out '.,?!'
    words = list(set(' '.join(lines).split()))

    vocab = {'[PAD]' : 0, '[CLS]' : 1, '[SEP]' : 2, '[MASK]' : 3}
   
    for idx, word in enumerate(words):
        vocab[word] = idx + 4

    vocab['[UNK]'] = len(vocab)
    print(vocab)

    with open('vocab.json', 'w') as f_vocab:
        json.dump(vocab, f_vocab)


def load_data(mode='train'):
    if mode == 'train':
        f_data = open('data/corpus.train.txt', 'rb')
        results = pickle.load(f_data)

        f_vocab = open('vocab.json')
        vocab = json.load(f_vocab)

        source = torch.tensor([result[0] for result in results])
        target = torch.tensor([result[1] for result in results])

        dataset = torch.utils.data.TensorDataset(source, target)

        train_size = int(0.8 * len(dataset))
        valid_size = len(dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(
            dataset, [train_size, valid_size])

        train_iter = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size)
        valid_iter = torch.utils.data.DataLoader(valid_dataset,
                                                batch_size=batch_size)
        return train_iter, valid_iter
    
    else:
        return test_iter


def zero_pad(x, segments, max_len):
    """
    Add zero padding tokens to input sentence
    """
    n_pad_tokens = max_len - len(x)
    x.extend([1] * n_pad_tokens)
    segments.extend([1] * n_pad_tokens)
    return x, segments