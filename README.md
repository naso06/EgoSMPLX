# Egocentric Whole-body Human Mesh Recovery with Prior-guided Learning 
<p align="center">
  <span style="display:inline-block; vertical-align:middle; margin-right:10px;">
    <img src="assets/CVLAB_logo.png" alt="Lab Logo" height="24">
  </span>
  <span style="display:inline-block; vertical-align:middle;">
    <b>Soyeon Na</b> &nbsp;&nbsp;
    <b>Seung Young Noh</b> &nbsp;&nbsp;
    <b>Ju Yong Chang</b>
  </span>
  
</p>

<p align="center">
  {naso06, kelvinnoh, jychang}@kw.ac.kr
  <br>
  Dept. of ECE, Kwangwoon University, Seoul, Korea
</p>


![teaser](assets/photo_2026-02-10_14-36-12.jpg)
## Installation

Install dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Configuration

Use the following config file for training:

- `config_ft_pesudogt.py`

## Pretrained Checkpoints

You can download pretrained checkpoints from Google Drive:

-  [Download Checkpoint](https://drive.google.com/drive/folders/1qhWqgy95NeWAGQnZ12a2PthyxciVHzx7?usp=drive_link)


## Testing 


```bash
sh slurm_test.sh <JOB_NAME> <NUM_GPU> <TRAIN_OUTPUT_DIR> <CKPT_ID>
```

**Arguments**

- `<JOB_NAME>`: SLURM job name
- `<NUM_GPU>`: number of GPUs to use
- `<TRAIN_OUTPUT_DIR>`: directory where training outputs are saved (e.g., `outputs/exp01`)
- `<CKPT_ID>`: checkpoint identifier to evaluate (e.g., a step number)

**Example**

```bash
sh slurm_test.sh exp01 4 outputs/exp01 10000
```
