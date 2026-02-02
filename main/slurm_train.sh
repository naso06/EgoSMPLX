#!/bin/bash
export PYTHONPATH=$(pwd)/..:$PYTHONPATH

JOB_NAME="smpler_x_h32"
GPUS=1
CONFIG="config_ft_pseudogtdataset.py"

python train.py \
    --num_gpus ${GPUS} \
    --exp_name output/train_${JOB_NAME} \
    --master_port $((RANDOM % 50 + 45600)) \
    --config ${CONFIG}

