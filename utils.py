import re
import json
from random import randrange, shuffle, random

import torch

torch.manual_seed(32)


def build_iter(params, mode='train'):
    """
    Build iterator for each mode
    """
    if mode == 'train':
        corpus = open('data/train.txt', 'r', encoding='utf-8')
        lines = ''.join(corpus.readlines())
        lines = re.sub('[.,!?\\-]', '', lines.lower()).split('\n')

        f_vocab = open('vocab.json')
        vocab = json.load(f_vocab)
        
        for idx, line in enumerate(lines):
            lines[idx] = [vocab[w] for w in line.split()]

        train_size = int(0.8 * len(lines))
        train_iter = lines[:train_size]
        valid_iter = lines[train_size:]

        train_iter = make_instance(train_iter, vocab, params)
        valid_iter = make_instance(valid_iter, vocab, params)
        
        return train_iter, valid_iter

    else:
        corpus = open('data/test.txt', 'r', encoding='utf-8')
        lines = ''.join(corpus.readlines())
        lines = re.sub('[.,!?\\-]', '', lines.lower()).split('\n')
        
        f_vocab = open('vocab.json')
        vocab = json.load(f_vocab)

        return make_instance(corpus, vocab, params)


def make_instance(corpus, vocab, params):
    """
    Make data instance with BERT configuration
    """
    batch = []
    positive = negative = 0

    while positive != params.bsz/2 or negative != params.bsz/2:
        sent_a_idx, sent_b_idx = randrange(len(corpus)), randrange(len(corpus))
        sent_a, sent_b = corpus[sent_a_idx], corpus[sent_b_idx]
        
        sentence = [vocab['[CLS]']] + sent_a + [vocab['[SEP]']] + sent_b + [vocab['[SEP]']]
        segments = [0] * (len(sent_a) + 2) + [1] * (len(sent_b) + 1)  # [CLS], [SEP], [SEP]

        n_pred = min(max(1, int(round(len(sentence) * 0.15))), 5)

        masked_sent, masked_idx = masked_lm(sentence, vocab, n_pred)
        padded_sent, padded_segments = zero_pad(masked_sent, segments, 30)

        if params.max_pred > n_pred:
            more_pad = 5 - n_pred
            masked_sent.extend([0] * more_pad)
            masked_idx.extend([0] * more_pad)
            
        if sent_a_idx + 1 == sent_b_idx and positive < params.bsz/2:
            batch.append([padded_sent, padded_segments, masked_sent, masked_idx, True])
            positive += 1
        elif sent_a_idx + 1 != sent_b_idx and negative < params.bsz/2:
            batch.append([padded_sent, padded_segments, masked_sent, masked_idx, False])
            negative += 1
    
    return batch


def masked_lm(sentence, vocab, n_pred):
    """
    Replace random tokens in sentence with [MASK] or other random tokens
    """
    idx_vocab = {i: w for i, w in enumerate(vocab)}
    
    # Minimum: 15% tokens of sentence, Maximum: params.max_predict
    masking_cand = [idx for idx, w in enumerate(sentence)
                    if w != vocab['[CLS]'] and w != vocab['[SEP]']]
    shuffle(masking_cand)

    masked_tokens, masked_idx = [], []
    for idx in masking_cand[:n_pred]:
        masked_idx.append(idx)
        masked_tokens.append(sentence[idx])
        
        if random() < 0.8:  # 80%
            sentence[idx] = vocab['[MASK]']
        elif random() < 0.5:  # 20% * 50% = 10%
            i = randrange(len(vocab))
            sentence[idx] = vocab[idx_vocab[i]]

    return sentence, masked_idx


def zero_pad(x, segments, max_len):
    """
    Add zero padding tokens to input sentence
    """
    n_pad_tokens = max_len - len(x)
    x.extend([0] * n_pad_tokens)
    segments.extend([0] * n_pad_tokens)
    return x, segments