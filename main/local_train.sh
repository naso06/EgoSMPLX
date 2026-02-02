#!/bin/bash
export PYTHONPATH=$(pwd)/..:$PYTHONPATH
unset SLURM_JOB_NAME
unset SLURM_NODEID
unset SLURM_PROCID
unset SLURM_NTASKS
unset SLURM_LOCALID
unset SLURM_NODELIST
unset SLURM_JOB_ID
unset SLURM_NNODES
unset RANK
unset WORLD_SIZE
unset MASTER_ADDR

JOB_NAME="smpler_x_b32"
GPUS=1
CONFIG="config_ft_pseudogtdataset.py"

python train.py \
    --num_gpus ${GPUS} \
    --exp_name /media/cv1/data/output/train_${JOB_NAME} \
    --master_port $((RANDOM % 50 + 45600)) \
    --config ${CONFIG}