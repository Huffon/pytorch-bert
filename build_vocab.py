import re
import json


def main():
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

    with open('vocab.json', 'w') as f_vocab:
        json.dump(vocab, f_vocab)


if __name__ == '__main__':
    main()
