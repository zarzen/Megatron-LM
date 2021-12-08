#!/bin/bash

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/fsx-dev/zhzhn/megatron-data/bench-bert-data_text_sentence
# VOCAB_FILE=<Specify path to vocab.txt>
VOCAB_FILE=/fsx-dev/zhzhn/vocabs/bert-large-uncased-vocab.txt
CHECKPOINT_PATH=/fsx-dev/zhzhn/megatron-LM-checkpoints

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_bert.py \
       --tensor-model-parallel-size 8 \
       --pipeline-model-parallel-size 1 \
       --num-layers 128 \
       --hidden-size 2560 \
       --num-attention-heads 40 \
       --micro-batch-size 8 \
       --global-batch-size 512 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --train-iters 10000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --min-lr 1.0e-5 \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 5 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --activations-checkpoint-method uniform \
       --no-bias-gelu-fusion \
       --no-bias-dropout-fusion \
       --no-masked-softmax-fusion \
       --fp16
