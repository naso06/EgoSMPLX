# Egocentric Whole-body Human Mesh Recovery with Prior-guided Learning


## Installation

Install dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt

Configuration

Use the following config file for training:

config_ft_pesudogt.py

When running training on a server, pass this file as {CONFIG_FILE}.



Training
Local Training

Run the local training script:

bash local_train.sh


This is useful for quick checks and debugging in a local environment.


Server Training (SLURM)

To train on a SLURM cluster, run:

sh slurm_train.sh {JOB_NAME} {NUM_GPU} {CONFIG_FILE}


Arguments

{JOB_NAME}: SLURM job name (e.g., exp01)

{NUM_GPU}: number of GPUs to use (e.g., 1, 4, 8)

{CONFIG_FILE}: path to the config file (e.g., config_ft_pesudogt.py)

Example

sh slurm_train.sh exp01 4 config_ft_pesudogt.py


Testing (SLURM)

To test on a SLURM cluster, run:

sh slurm_test.sh {JOB_NAME} {NUM_GPU} {TRAIN_OUTPUT_DIR} {CKPT_ID}


Arguments

{JOB_NAME}: SLURM job name

{NUM_GPU}: number of GPUs to use

{TRAIN_OUTPUT_DIR}: directory where training outputs are saved

{CKPT_ID}: checkpoint identifier to evaluate (e.g., a step number)

Example

sh slurm_test.sh exp01 4 outputs/exp01 10000

