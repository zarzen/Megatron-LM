#!/bin/bash

RANK=0
WORLD_SIZE=1
DATA_PATH=data/my-bert_text_sentence
CHECKPOINT_PATH=data/megatron-LM-checkpoints
VOCAB_FILE=data/bert-large-uncased-vocab.txt

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=34560

export DEBUG=1

python3 pretrain_bert.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 4 \
       --global-batch-size 8 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 10 \
       --lr-decay-iters 990000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --lr 0.0001 \
       --min-lr 0.00001 \
       --lr-decay-style linear \
       --lr-warmup-fraction .01 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --log-interval 10 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --activations-checkpoint-method uniform \
       --no-bias-gelu-fusion \
       --no-bias-dropout-fusion \
       --no-masked-softmax-fusion \
       --fp16
