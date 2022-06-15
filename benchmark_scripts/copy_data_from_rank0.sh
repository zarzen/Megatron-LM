#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

DATA_DIR=$SCRIPT_DIR/../data/

if [ -z "$1" ]
then
    echo "No rank0 IP is supplied"
else
    RANK0_IP=$1

    scp $RANK0_IP:~/Megatron-LM/data/* $DATA_DIR
fi
