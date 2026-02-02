#!/bin/bash

import os
import os.path as osp
# from types import SimpleNamespace
# use_token_decoder = True
# tokenizer_pretrained_model_path = '/home/cv1/works/SMPLer-X/pretrained_models/tokenizer.pth'
# token_cfg = '/home/cv1/works/SMPLer-X/tokenhmr/lib/configs_hydra/experiment/tokenhmr_release.yaml'
# token_decoder_ckpt_path = '/home/cv1/works/SMPLer-X/pretrained_models/tokenhmr_model_latest.ckpt'
# token_num = 160
# token_dim = 2048

use_dposerx = True
dposerx_cfg_path = 'configs.wholebody.subvp.mixed.get_config'
# egowholebody_flat_hand_mean = True

# id_adv_lambda = 0.5
# num_subjects = 7

fisheye_camera_path = '/home/cv1/works/SMPLer-X/main/fisheye.calibration_01_12.json'

no_aug = True
use_lora = False
use_smpl = False

# will be update in exp
num_gpus = 1
exp_name = 'output/exp1/egopw_finetuning'

# quick access
save_epoch = 1
lr = 1e-5
min_lr = 5e-7
end_epoch = 5
train_batch_size = 8
syncbn = True
bbox_ratio = 1.2

# continue
continue_train = True
start_over = True
pretrained_model_path = '/home/cv1/works/SMPLer-X/pretrained_models/smpler_x_h32.pth.tar'

# dataset setting
agora_fix_betas = True
agora_fix_global_orient_transl = True
agora_valid_root_pose = True

# for ubody ft
dataset_list = ['Human36M', 'MSCOCO', 'MPII', 'AGORA', 'EHF', 'SynBody', 'GTA_Human2', \
    'EgoBody_Egocentric', 'EgoBody_Kinect', 'UBody', 'PW3D', 'MuCo', 'PROX', 'PseudoGTDataset', 'Mo2Cap2', 'SceneEgo', 'EgoWholeBody', 'PseudoGTDataset_ext']
trainset_3d = [] 
trainset_2d = []
trainset_humandata = ['PseudoGTDataset'] 
testset = 'PseudoGTDataset'

use_cache = True

# strategy 
data_strategy = 'concat' # 'balance' need to define total_data_len
total_data_len = 'auto' # assign number or 'auto' for concat length

Talkshow_train_sample_interval = 10

# fine-tune
fine_tune = None # 'backbone', 'head', None for full network tuning

smplx_loss_weight = 1.0 #2 for agora_model for smplx shape
smplx_pose_weight = 10.0

smplx_kps_3d_weight = 100.0
smplx_kps_2d_weight = 1.0
token_loss_weight = 1.0
net_kps_2d_weight = 1.0

# id_loss_weight = 0.1

dposer_x_weight = 0.1
dposer_x_weight_min = 0.01
dposer_x_weight_warmup = 0 # epoch


agora_benchmark = 'agora_model' # 'agora_model', 'test_only'

model_type = 'smpler_x_h'
encoder_config_file = 'transformer_utils/configs/smpler_x/encoder/body_encoder_huge.py'
encoder_pretrained_model_path = '/home/cv1/works/SMPLer-X/pretrained_models/vitpose_huge.pth'
# feat_dim = 768
feat_dim = 1280 # vit_h

## =====FIXED ARGS============================================================
## model setting
upscale = 4
hand_pos_joint_num = 20
face_pos_joint_num = 72
num_task_token = 24
num_noise_sample = 0

## UBody setting
train_sample_interval = 10
test_sample_interval = 100
make_same_len = False

## input, output size
no_aug = True
input_img_shape = (1024, 1280) # H, W
input_body_shape = (256, 192)
output_hm_shape = (16, 16, 12)
input_hand_shape = (256, 256)
output_hand_hm_shape = (16, 16, 16)
output_face_hm_shape = (8, 8, 8)
input_face_shape = (192, 192)
focal = (5000, 5000)  # virtual focal lengths
princpt = (input_img_shape[1] / 2, input_img_shape[0] / 2)  # virtual principal point position
body_3d_size = 2
hand_3d_size = 0.3
face_3d_size = 0.3
camera_3d_size = 0.5

## training config
print_iters = 100
lr_mult = 1

## testing config
test_batch_size = 4

## others
num_thread = 0
vis = True

## directory
output_dir, model_dir, vis_dir, log_dir, result_dir, code_dir = None, None, None, None, None, None

use_latent_align = True
latent_align_weight = 1.0

