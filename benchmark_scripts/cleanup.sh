#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

CHECKPOINT_PATH="$SCRIPT_DIR/../data/megatron-LM-checkpoints"

if [ -d $CHECKPOINT_PATH ] 
then 
    echo "Has checkpiont files at $CHECKPOINT_PATH, deleting"
    rm -rf $CHECKPOINT_PATH
fi