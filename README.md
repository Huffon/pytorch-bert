# BERT PyTorch implementation

This repository contains unofficial [BERT](https://arxiv.org/abs/1810.04805) implmentation using PyTorch Framework.

## Usage

- To build vocabulary, run following code snippet

```bash
python build_vocab.py
```

- To pretrain BERT model, run following code snippet with options

```bash
python main.py \
    --mode MODE
    --max_len MAX_LEN
    --max_pred MAX_PRED
    --num_layers NUM_LAYERS
    --num_heads NUM_HEADS
    --num_segments NUM_SEGMENTS
    --hidden_dim HIDDEN_DIM
    --ffn_dim FFN_DIM
    --dropout DROPOUT
```

## TODO

- Finish `build_iter` and `make_instance` logic

## Reference

- [pytorchic-bert](https://github.com/dhlee347/pytorchic-bert)
- [nlp-tutorial](https://github.com/graykode/nlp-tutorial)
