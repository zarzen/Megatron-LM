#!/bin/bash

SEQ_LEN=512
MICRO_BS=8
GLOBAL_BS=1024
N_ITERS=10
GPUS_PER_NODE=8

N_LAYERS=12
HIDDEN_SIZE=2560
N_ATTEN_HEAD=40

# Change for multinode config
MASTER_ADDR=$1
MASTER_PORT=6000
NNODES=$2
NODE_RANK=$3
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
TMP=$4
PMP=$5

DATA_PATH=data/my-bert_text_sentence
# VOCAB_FILE=<Specify path to vocab.txt>
VOCAB_FILE=data/bert-large-uncased-vocab.txt
CHECKPOINT_PATH=data/megatron-LM-checkpoints

echo "NNODES $NNODES, NODE_RANK $NODE_RANK, WORLD_SIZE $WORLD_SIZE"
export NCCL_DEBUG=WARN
export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_bert.py \
       --tensor-model-parallel-size $TMP \
       --pipeline-model-parallel-size $PMP \
       --num-layers $N_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --num-attention-heads $N_ATTEN_HEAD \
       --micro-batch-size $MICRO_BS \
       --global-batch-size $GLOBAL_BS \
       --seq-length $SEQ_LEN \
       --max-position-embeddings $SEQ_LEN \
       --train-iters $N_ITERS \
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
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 1 \
       --activations-checkpoint-method uniform \
       --no-bias-gelu-fusion \
       --no-bias-dropout-fusion \
       --no-masked-softmax-fusion \
       --fp16
