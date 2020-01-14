import re
import json
from random import randrange, shuffle, random

import torch


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

        for idx, line in enumerate(lines):
            lines[idx] = [vocab[w] for w in line.split()]

        return make_instance(lines, vocab, params)


def make_instance(corpus, vocab, params):
    """
    Make data instance with BERT configuration
    """
    batch = []
    positive = negative = 0

    while positive != params.bsz/2 or negative != params.bsz/2:
        # Select random sentences from sentence list
        sent_a_idx, sent_b_idx = randrange(len(corpus)), randrange(len(corpus))
        sent_a, sent_b = corpus[sent_a_idx], corpus[sent_b_idx]
        
        # Concatenate randomly selected two sentences with [CLS] and [SEP] token
        input_ids = [vocab['[CLS]']] + sent_a + [vocab['[SEP]']] + sent_b + [vocab['[SEP]']]
        segment_ids = [0] * (len(sent_a) + 2) + [1] * (len(sent_b) + 1)  # ([CLS], [SEP]), [SEP]

        # Calculate the number of tokens to be masked
        n_pred = min(max(1, int(round(len(input_ids) * 0.15))), params.max_pred)

        # Conduct token replacement and zero padding
        masked_tokens, masked_pos = replace_token(input_ids, vocab, n_pred)
        padded_input_ids, padded_segment_ids = zero_pad(input_ids, segment_ids, params.max_len)

        # If max_pred is larger than n_pred, add padding tokens to masked_tokens and masked_pos
        if params.max_pred > n_pred:
            more_pad = params.max_pred - n_pred
            masked_tokens.extend([0] * more_pad)
            masked_pos.extend([0] * more_pad)
            
        if sent_a_idx + 1 == sent_b_idx and positive < params.bsz/2:
            batch.append([padded_input_ids, padded_segment_ids, masked_tokens, masked_pos, True])
            positive += 1
        elif sent_a_idx + 1 != sent_b_idx and negative < params.bsz/2:
            batch.append([padded_input_ids, padded_segment_ids, masked_tokens, masked_pos, False])
            negative += 1

    # Convert each input item into tensor and store them in device
    input_ids, segment_ids, masked_tokens, masked_pos, isNext = zip(*batch)
    input_ids, segment_ids, masked_tokens, masked_pos, isNext = \
        torch.LongTensor(input_ids).to(params.device), torch.LongTensor(segment_ids).to(params.device), \
            torch.LongTensor(masked_tokens).to(params.device), torch.LongTensor(masked_pos).to(params.device), torch.LongTensor(isNext).to(params.device)
    batch = [input_ids, segment_ids, masked_tokens, masked_pos, isNext]

    return batch


def replace_token(input_ids, vocab, n_pred):
    """
    Replace random tokens in sentence with [MASK] or other random tokens
    """
    idx_vocab = {i: w for i, w in enumerate(vocab)}
    
    # Minimum: 15% tokens of sentence, Maximum: params.max_predict
    masking_cands = [idx for idx, w in enumerate(input_ids)
                    if w != vocab['[CLS]'] and w != vocab['[SEP]']]
    shuffle(masking_cands)

    masked_tokens, masked_pos = [], []
    for pos in masking_cands[:n_pred]:
        masked_pos.append(pos)
        masked_tokens.append(input_ids[pos])
        
        if random() < 0.8:    # 80%: [MASK] replacement
            input_ids[pos] = vocab['[MASK]']
        elif random() < 0.5:  # 20% * 50% = 10%: random token replacement
            idx = randrange(len(vocab))
            input_ids[pos] = vocab[idx_vocab[idx]]

    return masked_tokens, masked_pos


def zero_pad(input_ids, segment_ids, max_len):
    """
    Add zero padding tokens to input sentence and segment
    """
    n_pad_tokens = max_len - len(input_ids)
    input_ids.extend([0] * n_pad_tokens)
    segment_ids.extend([0] * n_pad_tokens)
    return input_ids, segment_ids