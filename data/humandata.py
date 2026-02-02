import os
import os.path as osp
import numpy as np
import torch
import cv2
import json
import copy
from pycocotools.coco import COCO
from config import cfg
from utils.human_models import smpl_x
from utils.preprocessing import load_img, process_bbox, augmentation, process_db_coord, process_human_model_output, \
    get_fitting_error_3D
from utils.transforms import world2cam, cam2pixel, rigid_align
import tqdm
import time
from pdb import set_trace
import random
# import torch.nn as nn
from smplx import SMPLX
from main.FishEyeCalibrated import FishEyeCameraCalibrated
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.nn import functional as F
from scipy.spatial import cKDTree


import os
import cv2
import numpy as np

# ---- smplx_joint_img / smplx_joint_proj 는 아래 joints_name 순서를 따른다고 가정 ----
# body(25): 0~24
# left hand(20): 25~44
# right hand(20): 45~64

BODY_HANDS_NUM = 65  # 25 + 20 + 20

# 손가락 체인(각 4개) : base->...->tip
# left hand indices in this representation:
# L_Thumb_1..4: 25,26,27,28
# L_Index_1..4: 29,30,31,32
# L_Middle_1..4:33,34,35,36
# L_Ring_1..4: 37,38,39,40
# L_Pinky_1..4:41,42,43,44
#
# right hand:
# R_Thumb_1..4:45,46,47,48
# R_Index_1..4:49,50,51,52
# R_Middle_1..4:53,54,55,56
# R_Ring_1..4: 57,58,59,60
# R_Pinky_1..4:61,62,63,64

def build_body_hands_edges():
    edges = []

    # ---- Body edges (0~24) ----
    # body joint names (0~24):
    # 0 Pelvis
    # 1 L_Hip, 2 R_Hip
    # 3 L_Knee, 4 R_Knee
    # 5 L_Ankle, 6 R_Ankle
    # 7 Neck
    # 8 L_Shoulder, 9 R_Shoulder
    # 10 L_Elbow, 11 R_Elbow
    # 12 L_Wrist, 13 R_Wrist
    # 14 L_Big_toe, 15 L_Small_toe, 16 L_Heel
    # 17 R_Big_toe, 18 R_Small_toe, 19 R_Heel
    # 20 L_Ear, 21 R_Ear, 22 L_Eye, 23 R_Eye, 24 Nose

    # pelvis to hips
    edges += [(0, 1), (0, 2)]
    # legs
    edges += [(1, 3), (3, 5)]
    edges += [(2, 4), (4, 6)]
    # ankles to feet (simple, robust)
    edges += [(5, 16), (6, 19)]  # ankle -> heel
    edges += [(5, 14), (5, 15)]  # ankle -> toes (optional but ok)
    edges += [(6, 17), (6, 18)]

    # torso/shoulders (SMPLX 55에는 spine이 있지만 여기서는 neck만 있으므로 pelvis->neck은 "가상" 연결)
    # 시각화 목적이면 pelvis->neck 연결을 넣어도 좋습니다.
    edges += [(0, 7)]

    # arms
    edges += [(7, 8), (8, 10), (10, 12)]
    edges += [(7, 9), (9, 11), (11, 13)]

    # # head/face 최소 연결(원하면 제거 가능)
    # edges += [(7, 24)]          # neck -> nose
    # edges += [(24, 22), (24, 23)]  # nose -> eyes
    # edges += [(22, 20), (23, 21)]  # eyes -> ears

    # ---- Hand edges ----
    # 손목(각각 body의 L_Wrist=12, R_Wrist=13)과 hand root(Thumb_1 등)를 연결
    # Left: connect wrist(12) to each finger base(Thumb_1=25, Index_1=29, Middle_1=33, Ring_1=37, Pinky_1=41)
    edges += [(12, 25), (12, 29), (12, 33), (12, 37), (12, 41)]
    # Right: wrist(13) to finger bases
    edges += [(13, 45), (13, 49), (13, 53), (13, 57), (13, 61)]

    def chain(a, b, c, d):
        return [(a, b), (b, c), (c, d)]

    # Left finger chains
    edges += chain(25, 26, 27, 28)  # thumb
    edges += chain(29, 30, 31, 32)  # index
    edges += chain(33, 34, 35, 36)  # middle
    edges += chain(37, 38, 39, 40)  # ring
    edges += chain(41, 42, 43, 44)  # pinky

    # Right finger chains
    edges += chain(45, 46, 47, 48)
    edges += chain(49, 50, 51, 52)
    edges += chain(53, 54, 55, 56)
    edges += chain(57, 58, 59, 60)
    edges += chain(61, 62, 63, 64)

    return edges

BODY_HANDS_EDGES = build_body_hands_edges()


def _to_numpy_xy(joints):
    """
    joints: (J,2) or (J,3) or torch.Tensor
    return: (J,2) float32 numpy
    """
    if joints is None:
        return None
    if isinstance(joints, np.ndarray):
        arr = joints
    else:
        # torch tensor or list
        try:
            import torch
            if isinstance(joints, torch.Tensor):
                arr = joints.detach().cpu().numpy()
            else:
                arr = np.asarray(joints)
        except Exception:
            arr = np.asarray(joints)
    arr = arr.astype(np.float32)
    if arr.ndim != 2:
        arr = arr.reshape(-1, arr.shape[-1])
    return arr[:, :2].copy()

def _apply_homography_xy(xy, H):
    """
    xy: (J,2)
    H: (3,3) (e.g., bb2img_trans)
    """
    if xy is None or H is None:
        return xy
    xy1 = np.concatenate([xy, np.ones((xy.shape[0], 1), dtype=np.float32)], axis=1)  # (J,3)
    out = (H @ xy1.T).T
    out = out[:, :2] / (out[:, 2:3] + 1e-8)
    return out.astype(np.float32)

def draw_2d_pose_overlay(
    img_bgr,
    joints_gt,
    joints_pred,
    valid_gt=None,
    valid_pred=None,
    edges=None,
    subset_idx=None,  # 추가: 예) np.arange(65)
    gt_color=(0, 255, 0),
    pred_color=(0, 0, 255),
    radius=3,
    thickness=2,
    draw_index=False
):
    out = img_bgr.copy()

    gt = _to_numpy_xy(joints_gt)
    pr = _to_numpy_xy(joints_pred)
    if gt is None or pr is None:
        return out

    J = min(gt.shape[0], pr.shape[0])
    gt, pr = gt[:J], pr[:J]

    def _valid_mask(v, J):
        if v is None:
            return np.ones((J,), dtype=np.uint8)
        v = np.asarray(v).reshape(-1)
        if v.shape[0] != J:
            return np.ones((J,), dtype=np.uint8)
        return (v > 0).astype(np.uint8)

    vgt = _valid_mask(valid_gt, J)
    vpr = _valid_mask(valid_pred, J)

    if subset_idx is None:
        subset_mask = np.ones((J,), dtype=np.uint8)
    else:
        subset_mask = np.zeros((J,), dtype=np.uint8)
        subset_idx = np.asarray(subset_idx).reshape(-1)
        subset_idx = subset_idx[(subset_idx >= 0) & (subset_idx < J)]
        subset_mask[subset_idx] = 1

    # edges 그리기
    if edges is not None:
        for (i, j) in edges:
            if i < J and j < J:
                if subset_mask[i] and subset_mask[j]:
                    if vgt[i] and vgt[j]:
                        p1 = tuple(np.round(gt[i]).astype(int))
                        p2 = tuple(np.round(gt[j]).astype(int))
                        cv2.line(out, p1, p2, gt_color, thickness, lineType=cv2.LINE_AA)
                    if vpr[i] and vpr[j]:
                        p1 = tuple(np.round(pr[i]).astype(int))
                        p2 = tuple(np.round(pr[j]).astype(int))
                        cv2.line(out, p1, p2, pred_color, thickness, lineType=cv2.LINE_AA)

    # 점 그리기
    for i in range(J):
        if not subset_mask[i]:
            continue
        if vgt[i] and np.isfinite(gt[i]).all():
            p = tuple(np.round(gt[i]).astype(int))
            cv2.circle(out, p, radius, gt_color, -1, lineType=cv2.LINE_AA)
            if draw_index:
                cv2.putText(out, str(i), (p[0] + 4, p[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, gt_color, 1, cv2.LINE_AA)

        if vpr[i] and np.isfinite(pr[i]).all():
            p = tuple(np.round(pr[i]).astype(int))
            cv2.circle(out, p, radius, pred_color, -1, lineType=cv2.LINE_AA)
            if draw_index:
                cv2.putText(out, str(i), (p[0] + 4, p[1] + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, pred_color, 1, cv2.LINE_AA)

    return out

def maybe_scale_from_hm_to_img(joints_xy, img_w, img_h, hm_w, hm_h):
    """
    joints_xy가 (0~hm_w/hm_h) 범위라면 img 크기로 스케일.
    """
    if joints_xy is None:
        return None
    xy = _to_numpy_xy(joints_xy)
    if xy is None:
        return None
    
    
    xy[:, 0] = xy[:, 0] / float(hm_w) * float(img_w)
    xy[:, 1] = xy[:, 1] / float(hm_h) * float(img_h)
    return xy



# from smplx.lbs import batch_rodrigues  # smplx에서 제공
# from pytorch3d.transforms import axis_angle_to_matrix
# SMPLX_MODEL_DIR = '/home/cv1/works/smplify-x/models/smplx'
# class SMPLXHead(nn.Module):
#     def __init__(self, focal_length=5000., img_res=224):
#         super(SMPLXHead, self).__init__()
#         self.smpl = SMPLX(SMPLX_MODEL_DIR, pose2rot=False,
#             create_jaw_pose=False,
#             create_leye_pose=False,
#             create_reye_pose=False,
#             create_expression=False,
#             create_left_hand_pose=True,
#             create_right_hand_pose=True,
#             create_transl=False,)
#         self.add_module('smpl', self.smpl)
#         self.focal_length = focal_length
#         self.img_res = img_res

#     def forward(self, global_orient, body_pose, betas, cam=None, normalize_joints2d=False):
#         '''
#         :param rotmat: rotation in euler angles format (N,J,3,3)
#         :param shape: smpl betas
#         :param cam: weak perspective camera
#         :param normalize_joints2d: bool, normalize joints between -1, 1 if true
#         :return: dict with keys 'vertices', 'joints3d', 'joints2d' if cam is True
#         '''
#         smpl_output = self.smpl(
#             betas=betas,
#             body_pose=body_pose.contiguous(),
#             global_orient=global_orient.contiguous(),
           
#         )
#         # smpl_output = self.smpl(
#         #     betas=shape,
#         #     body_pose=rotmat[:, 1:].contiguous(),
#         #     pose2rot=False,
#         # )

#         output = {
#             'smpl_betas' : betas,
#             'smpl_thetas' : torch.cat([global_orient,body_pose], dim=1),
#             'smpl_vertices': smpl_output.vertices,
#             'smpl_joints3d': smpl_output.joints,
#         }

#         return smpl_output,output
    
KPS2D_KEYS = ['keypoints2d', 'keypoints2d_smplx', 'keypoints2d_smpl', 'keypoints2d_original']
KPS3D_KEYS = ['keypoints3d_cam', 'keypoints3d', 'keypoints3d_smplx','keypoints3d_smpl' ,'keypoints3d_original'] 
# keypoints3d_cam with root-align has higher priority, followed by old version key keypoints3d
# when there is keypoints3d_smplx, use this rather than keypoints3d_original

POS_JOINT_NAMES = smpl_x.pos_joints_name  # ('Pelvis', 'L_Hip', ... 'R_Pinky_4')

# SMPL-X 55 joints order (matches your screenshot)



hands_meanr = np.array([ 0.11167871, -0.04289218,  0.41644183,  0.10881133,  0.06598568,
        0.75622   , -0.09639297,  0.09091566,  0.18845929, -0.11809504,
       -0.05094385,  0.5295845 , -0.14369841, -0.0552417 ,  0.7048571 ,
       -0.01918292,  0.09233685,  0.3379135 , -0.45703298,  0.19628395,
        0.6254575 , -0.21465237,  0.06599829,  0.50689423, -0.36972436,
        0.06034463,  0.07949023, -0.1418697 ,  0.08585263,  0.63552827,
       -0.3033416 ,  0.05788098,  0.6313892 , -0.17612089,  0.13209307,
        0.37335458,  0.8509643 , -0.27692273,  0.09154807, -0.49983943,
       -0.02655647, -0.05288088,  0.5355592 , -0.04596104,  0.27735803]).reshape(15, -1)
hands_meanl = np.array([ 0.11167871,  0.04289218, -0.41644183,  0.10881133, -0.06598568,
       -0.75622   , -0.09639297, -0.09091566, -0.18845929, -0.11809504,
        0.05094385, -0.5295845 , -0.14369841,  0.0552417 , -0.7048571 ,
       -0.01918292, -0.09233685, -0.3379135 , -0.45703298, -0.19628395,
       -0.6254575 , -0.21465237, -0.06599829, -0.50689423, -0.36972436,
       -0.06034463, -0.07949023, -0.1418697 , -0.08585263, -0.63552827,
       -0.3033416 , -0.05788098, -0.6313892 , -0.17612089, -0.13209307,
       -0.37335458,  0.8509643 ,  0.27692273, -0.09154807, -0.49983943,
        0.02655647,  0.05288088,  0.5355592 ,  0.04596104, -0.27735803]).reshape(15, -1)






mo2cap2_chain = [
    [0, 1, 2, 3],
    [0, 4, 5, 6],
    [1, 7, 8, 9, 10],
    [4, 11, 12, 13, 14],
    [7, 11],
]

mo2cap2_to_smplx_idx = [
    12,  # neck
    17,  # right_shoulder
    19,  # right_elbow
    21,  # right_wrist
    16,  # left_shoulder
    18,  # left_elbow
    20,  # left_wrist
    2,   # right_hip
    5,   # right_knee
    8,   # right_ankle
    11,  # right_foot
    1,   # left_hip
    4,   # left_knee
    7,   # left_ankle
    10   # left_foot
]

MO2CAP2_PARTS = {
    'right_arm': [0, 1, 2, 3],
    'left_arm': [4, 5, 6],
    'right_leg': [7, 8, 9, 10],
    'left_leg': [11, 12, 13, 14]
}



mo2cap2_to_smplx = {
    "neck": "Neck",
    "right_shoulder": "R_Shoulder",
    "right_elbow": "R_Elbow",
    "right_wrist": "R_Wrist",
    "left_shoulder": "L_Shoulder",
    "left_elbow": "L_Elbow",
    "left_wrist": "L_Wrist",
    "right_hip": "R_Hip",
    "right_knee": "R_Knee",
    "right_ankle": "R_Ankle",
    "right_foot": "R_Heel",
    "left_hip": "L_Hip",
    "left_knee": "L_Knee",
    "left_ankle": "L_Ankle",
    "left_foot": "L_Heel",
}


mo2cap2_joint_names = [
    "neck",             # 
    "right_shoulder",   # 
    "right_elbow",      # 
    "right_wrist",      # 
    "left_shoulder",    # 
    "left_elbow",       # 
    "left_wrist",       # 
    "right_hip",        # 
    "right_knee",       #
    "right_ankle",      #
    "right_foot",       # 
    "left_hip",         #
    "left_knee",        #
    "left_ankle",       #
    "left_foot"         #
]


# SMPL-X 55 joints order (matches your screenshot)
SMPLX_JOINT_LIST = [
    # body joint (0 ~ 21)
    'pelvis', 'left_hip', 'right_hip', 'spine1',
    'left_knee', 'right_knee', 'spine2',
    'left_ankle', 'right_ankle', 'spine3',
    'left_foot', 'right_foot', 'neck',
    'left_collar', 'right_collar', 'head',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',

    # jaw joint (22)
    'jaw',

    # eye joint (23, 24)
    'left_eye_smplhf', 'right_eye_smplhf',

    # left hand joint (25 ~ 39)
    'left_index1', 'left_index2', 'left_index3',
    'left_middle1', 'left_middle2', 'left_middle3',
    'left_pinky1', 'left_pinky2', 'left_pinky3',
    'left_ring1', 'left_ring2', 'left_ring3',
    'left_thumb1', 'left_thumb2', 'left_thumb3',

    # right hand joint (40 ~ 54)
    'right_index1', 'right_index2', 'right_index3',
    'right_middle1', 'right_middle2', 'right_middle3',
    'right_pinky1', 'right_pinky2', 'right_pinky3',
    'right_ring1', 'right_ring2', 'right_ring3',
    'right_thumb1', 'right_thumb2', 'right_thumb3',
]


smplx_joints_name_137 = [
    'Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Neck',
    'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',
    'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel',
    'L_Ear', 'R_Ear', 'L_Eye', 'R_Eye', 'Nose',
    'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4',
    'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4',
    'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4',
    'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4',
    'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4',
    'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4',
    'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4',
    'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4',
    'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4',
    'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4',
    *['Face_' + str(i) for i in range(1, 73)]
]

MO2CAP2_TO_POS_IDX = [
    POS_JOINT_NAMES.index(mo2cap2_to_smplx[j_name])
    for j_name in mo2cap2_joint_names
]


class Cache():
    """ A custom implementation for SMPLer_X pipeline
        Need to run tool/cache/fix_cache.py to fix paths
    """
    def __init__(self, load_path=None):
        if load_path is not None:
            self.load(load_path)

    def load(self, load_path):
        self.load_path = load_path
        self.cache = np.load(load_path, allow_pickle=True)
        self.data_len = self.cache['data_len']
        self.data_strategy = self.cache['data_strategy']
        assert self.data_len == len(self.cache) - 2  # data_len, data_strategy
        self.cache = None

    @classmethod
    def save(cls, save_path, data_list, data_strategy):
        assert save_path is not None, 'save_path is None'
        data_len = len(data_list)
        cache = {}
        for i, data in enumerate(data_list):
            cache[str(i)] = data
        assert len(cache) == data_len
        # update meta
        cache.update({
            'data_len': data_len,
            'data_strategy': data_strategy})

        np.savez_compressed(save_path, **cache)
        print(f'Cache saved to {save_path}.')

    # def shuffle(self):
    #     random.shuffle(self.mapping)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        if self.cache is None:
            self.cache = np.load(self.load_path, allow_pickle=True)
        # mapped_idx = self.mapping[idx]
        # cache_data = self.cache[str(mapped_idx)]
        cache_data = self.cache[str(idx)]
        data = cache_data.item()
        return data


class HumanDataset(torch.utils.data.Dataset):

    # same mapping for 144->137 and 190->137
    SMPLX_137_MAPPING = [
        0, 1, 2, 4, 5, 7, 8, 12, 16, 17, 18, 19, 20, 21, 60, 61, 62, 63, 64, 65, 59, 58, 57, 56, 55, 37, 38, 39, 66,
        25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31, 32, 33, 70, 52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45,
        73, 49, 50, 51, 74, 46, 47, 48, 75, 22, 15, 56, 57, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
        90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
        114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
        136, 137, 138, 139, 140, 141, 142, 143]

    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split

        # dataset information, to be filled by child class
        self.img_dir = None
        self.annot_path = None
        self.annot_path_cache = None
        self.use_cache = False
        self.save_idx = 0
        self.img_shape = None  # (h, w)
        self.cam_param = None  # {'focal_length': (fx, fy), 'princpt': (cx, cy)}
        self.use_betas_neutral = False

        self.joint_set = {
            'joint_num': smpl_x.joint_num,
            'joints_name': smpl_x.joints_name,
            'flip_pairs': smpl_x.flip_pairs}
        self.joint_set['root_joint_idx'] = self.joint_set['joints_name'].index('Pelvis')

        self.undist_w, self.undist_h = 1280, 1024
        fov_deg = 120
        fov_rad = np.deg2rad(fov_deg)

        x = np.linspace(-1, 1, self.undist_w)
        y = np.linspace(-1, 1, self.undist_h)
        xx, yy = np.meshgrid(x, y)
        z = 1.0 / np.tan(fov_rad / 2)
        rays = np.stack([xx, yy, np.full_like(xx, z)], axis=-1)
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        rays_flat = rays.reshape(-1, 3)

        calib_path = "/home/cv1/works/SMPLer-X/main/fisheye.calibration_01_12.json"
        self.fisheye_camera = FishEyeCameraCalibrated(calib_path)

        # Only once: build pixel map
        mapped_pixels = []
        for r in rays_flat:
            try:
                uv = self.fisheye_camera.world2camera(np.array([r]))
                mapped_pixels.append(uv[0])
            except:
                mapped_pixels.append([-1, -1])

        self.mapped_pixels = np.array(mapped_pixels, dtype=np.float32).reshape(self.undist_h, self.undist_w, 2)

    def load_cache(self, annot_path_cache):
        datalist = Cache(annot_path_cache)
        assert datalist.data_strategy == getattr(cfg, 'data_strategy', None), \
            f'Cache data strategy {datalist.data_strategy} does not match current data strategy ' \
            f'{getattr(cfg, "data_strategy", None)}'
        return datalist

    def save_cache(self, annot_path_cache, datalist):
        print(f'[{self.__class__.__name__}] Caching datalist to {self.annot_path_cache}...')
        Cache.save(
            annot_path_cache,
            datalist,
            data_strategy=getattr(cfg, 'data_strategy', None)
        )

    def _get_subject_scene_dir(self, img_path):
        """
        예)
          dataset/train/S01/SceneA/imgs/img_000001.jpg
          dataset/train/S01/SceneA/img_000001.jpg
          test/S01/SceneB/imgs/xxx.png
        같은 경로에서 {subject}/{scene} (= S01/SceneA, S01/SceneB)만 떼서 반환.
        """
        if not isinstance(img_path, str):
            return ""

        path = img_path.replace("\\", "/")

        # self.img_dir 기준으로 불필요한 상위폴더 제거
        if getattr(self, "img_dir", None):
            try:
                path = os.path.relpath(path, start=self.img_dir).replace("\\", "/")
            except Exception:
                pass

        parts = [p for p in path.split("/") if p]
        if len(parts) <= 1:
            return ""

        # 마지막은 파일 이름이므로 빼고 "디렉토리들"만 남김
        dir_parts = parts[:-1]
        if not dir_parts:
            return ""

        # 마지막 디렉토리가 imgs 이면 제거 (dataset/.../{subject}/{scene}/imgs/파일 구조 대응)
        if dir_parts[-1].lower() in ("imgs", "img"):
            dir_parts = dir_parts[:-1]

        # 이제 남은 디렉토리 중 끝에서 두 개를 {subject}/{scene}으로 사용
        if len(dir_parts) >= 2:
            return os.path.join(dir_parts[-2], dir_parts[-1])

        return ""

    def visualize_joints(self, img, joint_img, joint_valid=None, figsize=(10, 10)):
        import matplotlib.pyplot as plt
        import numpy as np

        if isinstance(img, torch.Tensor):
            img = img.clone().detach().cpu()
            if img.min() < 0:  # 이미 Normalize 된 경우
                img = (img * 0.5 + 0.5)  # 예를 들어 mean=0.5, std=0.5 로 normalize 한 경우
            img = img.permute(1, 2, 0).numpy()  # (H, W, C)

        img = (img * 255).clip(0, 255).astype(np.uint8)

        plt.figure(figsize=figsize)
        plt.imshow(img)
        
        for i, joint in enumerate(joint_img):
            x, y, z = joint
            if joint_valid is not None:
                if joint_valid[i] == 0:
                    continue
            plt.scatter(x, y, c='r', s=20)
            plt.text(x, y, str(i), fontsize=8, color='yellow')
        
        plt.axis('off')
        plt.show()

    def load_data(self, train_sample_interval=1, test_sample_interval=1):

        content = np.load(self.annot_path, allow_pickle=True)
        num_examples = len(content['image_path']) 

        if 'meta' in content:
            meta = content['meta'].item()
            print('meta keys:', meta.keys())
        else:
            meta = None
            print('No meta info provided! Please give height and width manually')

        print(f'Start loading humandata {self.annot_path} into memory...\nDataset includes: {content.files}'); tic = time.time()
        image_path = content['image_path']

        if meta is not None and 'height' in meta:
            height = np.array(meta['height'])
            width = np.array(meta['width'])
            image_shape = np.stack([height, width], axis=-1)
        else:
            image_shape = None

        bbox_xywh = content['bbox_xywh']

        if 'smplx' in content:
            smplx = content['smplx'].item()
            as_smplx = 'smplx'
        elif 'smpl' in content:
            smplx = content['smpl'].item()
            as_smplx = 'smpl'
        elif 'smplh' in content:
            smplx = content['smplh'].item()
            as_smplx = 'smplh'

        # TODO: temp solution, should be more general. But SHAPY is very special
        elif self.__class__.__name__ == 'SHAPY':
            smplx = {}

        else:
            raise KeyError('No SMPL for SMPLX available, please check keys:\n'
                        f'{content.files}')

        print('Smplx param', smplx.keys())

        if 'lhand_bbox_xywh' in content and 'rhand_bbox_xywh' in content:
            lhand_bbox_xywh = content['lhand_bbox_xywh']
            rhand_bbox_xywh = content['rhand_bbox_xywh']
        else:
            lhand_bbox_xywh = np.zeros_like(bbox_xywh)
            rhand_bbox_xywh = np.zeros_like(bbox_xywh)

        if 'face_bbox_xywh' in content:
            face_bbox_xywh = content['face_bbox_xywh']
        else:
            face_bbox_xywh = np.zeros_like(bbox_xywh)

        decompressed = False
        if content['__keypoints_compressed__']:
            decompressed_kps = self.decompress_keypoints(content)
            decompressed = True

        keypoints3d = None
        valid_kps3d = False
        keypoints3d_mask = None
        valid_kps3d_mask = False
        for kps3d_key in KPS3D_KEYS:
            if kps3d_key in content:
                keypoints3d = decompressed_kps[kps3d_key][:, self.SMPLX_137_MAPPING, :3] if decompressed \
                else content[kps3d_key][:, self.SMPLX_137_MAPPING, :3]
                valid_kps3d = True

                if f'{kps3d_key}_mask' in content:
                    keypoints3d_mask = content[f'{kps3d_key}_mask'][self.SMPLX_137_MAPPING]
                    valid_kps3d_mask = True
                elif 'keypoints3d_mask' in content:
                    keypoints3d_mask = content['keypoints3d_mask'][self.SMPLX_137_MAPPING]
                    valid_kps3d_mask = True
                break

        for kps2d_key in KPS2D_KEYS:
            if kps2d_key in content:
                keypoints2d = decompressed_kps[kps2d_key][:, self.SMPLX_137_MAPPING, :2] if decompressed \
                    else content[kps2d_key][:, self.SMPLX_137_MAPPING, :2]

                if f'{kps2d_key}_mask' in content:
                    keypoints2d_mask = content[f'{kps2d_key}_mask'][self.SMPLX_137_MAPPING]
                elif 'keypoints2d_mask' in content:
                    keypoints2d_mask = content['keypoints2d_mask'][self.SMPLX_137_MAPPING]
                break

        mask = keypoints3d_mask if valid_kps3d_mask \
                else keypoints2d_mask

        print('Done. Time: {:.2f}s'.format(time.time() - tic))

        datalist = []
        for i in tqdm.tqdm(range(int(num_examples))):
            if self.data_split == 'train' and i % train_sample_interval != 0:
                continue
            if self.data_split == 'test' and i % test_sample_interval != 0:
                continue
            img_path = osp.join(self.img_dir, image_path[i])

            # ====== 추가: 0byte / 존재하지 않는 이미지 스킵 ======
            if (not osp.exists(img_path)) or (osp.getsize(img_path) == 0):
                print(f'[{self.__class__.__name__}] Skip invalid image: {img_path}')
                continue
            # ===============================================

            img_shape = image_shape[i] if image_shape is not None else self.img_shape

            bbox = bbox_xywh[i][:4]

            if hasattr(cfg, 'bbox_ratio'):
                bbox_ratio = cfg.bbox_ratio * 0.833 # preprocess body bbox is giving 1.2 box padding
            else:
                bbox_ratio = 1.25
            bbox = process_bbox(bbox, img_width=img_shape[1], img_height=img_shape[0], ratio=bbox_ratio)
            if bbox is None: continue

            # hand/face bbox
            lhand_bbox = lhand_bbox_xywh[i]
            rhand_bbox = rhand_bbox_xywh[i]
            face_bbox = face_bbox_xywh[i]

            if lhand_bbox[-1] > 0:  # conf > 0
                lhand_bbox = lhand_bbox[:4]
                if hasattr(cfg, 'bbox_ratio'):
                    lhand_bbox = process_bbox(lhand_bbox, img_width=img_shape[1], img_height=img_shape[0], ratio=cfg.bbox_ratio)
                if lhand_bbox is not None:
                    lhand_bbox[2:] += lhand_bbox[:2]  # xywh -> xyxy
            else:
                lhand_bbox = None
            if rhand_bbox[-1] > 0:
                rhand_bbox = rhand_bbox[:4]
                if hasattr(cfg, 'bbox_ratio'):
                    rhand_bbox = process_bbox(rhand_bbox, img_width=img_shape[1], img_height=img_shape[0], ratio=cfg.bbox_ratio)
                if rhand_bbox is not None:
                    rhand_bbox[2:] += rhand_bbox[:2]  # xywh -> xyxy
            else:
                rhand_bbox = None
            if face_bbox[-1] > 0:
                face_bbox = face_bbox[:4]
                if hasattr(cfg, 'bbox_ratio'):
                    face_bbox = process_bbox(face_bbox, img_width=img_shape[1], img_height=img_shape[0], ratio=cfg.bbox_ratio)
                if face_bbox is not None:
                    face_bbox[2:] += face_bbox[:2]  # xywh -> xyxy
            else:
                face_bbox = None

            joint_img = keypoints2d[i]
            joint_valid = mask.reshape(-1, 1)
            # num_joints = joint_cam.shape[0]
            # joint_valid = np.ones((num_joints, 1))
            if valid_kps3d:
                joint_cam = keypoints3d[i]
            else:
                joint_cam = None

            smplx_param = {k: v[i] for k, v in smplx.items()}

            smplx_param['root_pose'] = smplx_param.pop('global_orient', np.zeros(3))
            smplx_param['shape'] = smplx_param.pop('betas', None)
            smplx_param['trans'] = smplx_param.pop('transl', np.zeros(3))
            smplx_param['lhand_pose'] = smplx_param.pop('left_hand_pose', None)
            smplx_param['rhand_pose'] = smplx_param.pop('right_hand_pose', None)
            smplx_param['expr'] = smplx_param.pop('expression', None)

            if smplx_param['lhand_pose'] is None:
                smplx_param['lhand_valid'] = False
            else:
                smplx_param['lhand_valid'] = True
            if smplx_param['rhand_pose'] is None:
                smplx_param['rhand_valid'] = False
            else:
                smplx_param['rhand_valid'] = True
            if smplx_param['expr'] is None:
                smplx_param['face_valid'] = False
            else:
                smplx_param['face_valid'] = True

            # TODO do not fix betas, give up shape supervision
            if 'betas_neutral' in smplx_param:
                smplx_param['shape'] = smplx_param.pop('betas_neutral')

            # TODO fix shape of poses
            if self.__class__.__name__ == 'Talkshow':
                smplx_param['body_pose'] = smplx_param['body_pose'].reshape(21, 3)
                smplx_param['lhand_pose'] = smplx_param['lhand_pose'].reshape(15, 3)
                smplx_param['rhand_pose'] = smplx_param['lhand_pose'].reshape(15, 3)
                smplx_param['expr'] = smplx_param['expr'][:10]

            if self.__class__.__name__ == 'BEDLAM':
                smplx_param['shape'] = smplx_param['shape'][:10]
                # manually set flat_hand_mean = True
                smplx_param['lhand_pose'] -= hands_meanl
                smplx_param['rhand_pose'] -= hands_meanr


            if as_smplx == 'smpl':
                smplx_param['shape'] = np.zeros(10, dtype=np.float32) # drop smpl betas for smplx
                smplx_param['body_pose'] = smplx_param['body_pose'][:21, :] # use smpl body_pose on smplx

            if as_smplx == 'smplh':
                smplx_param['shape'] = np.zeros(10, dtype=np.float32) # drop smpl betas for smplx

            if smplx_param['lhand_pose'] is None:
                smplx_param['lhand_valid'] = False
            else:
                smplx_param['lhand_valid'] = True
            if smplx_param['rhand_pose'] is None:
                smplx_param['rhand_valid'] = False
            else:
                smplx_param['rhand_valid'] = True
            if smplx_param['expr'] is None:
                smplx_param['face_valid'] = False
            else:
                smplx_param['face_valid'] = True

            if joint_cam is not None and np.any(np.isnan(joint_cam)):
                continue

            datalist.append({
                'img_path': img_path,
                'img_shape': img_shape,
                'bbox': bbox,
                'lhand_bbox': lhand_bbox,
                'rhand_bbox': rhand_bbox,
                'face_bbox': face_bbox,
                'joint_img': joint_img,
                'joint_cam': joint_cam,
                'joint_valid': joint_valid,
                'smplx_param': smplx_param,
                'smplx': smplx})

        # save memory
        del content, image_path, bbox_xywh, lhand_bbox_xywh, rhand_bbox_xywh, face_bbox_xywh, keypoints3d, keypoints2d

        if self.data_split == 'train':
            print(f'[{self.__class__.__name__} train] original size:', int(num_examples),
                  '. Sample interval:', train_sample_interval,
                  '. Sampled size:', len(datalist))

        if (getattr(cfg, 'data_strategy', None) == 'balance' and self.data_split == 'train') or \
                getattr(cfg, 'eval_on_train', False):
            print(f'[{self.__class__.__name__}] Using [balance] strategy with datalist shuffled...')
            random.seed(2023)
            random.shuffle(datalist)

            if getattr(cfg, 'eval_on_train', False):
                return datalist

        return datalist 
    

    def __len__(self):
        return len(self.datalist)
    
    def undistort_keypoints_from_fisheye(self, joints_2d, fisheye_camera, undist_w=1280, undist_h=1024, fov_deg=120):
        """
        joints_2d: (N, 2) ndarray or torch.Tensor, fisheye 이미지 기준 2D 좌표
        fisheye_camera: FishEyeCameraCalibrated 객체
        Return: 보정된 이미지 기준 2D 좌표 (N, 2)
        """
        # 입력이 torch.Tensor인 경우 numpy로 변환
        is_tensor = isinstance(joints_2d, torch.Tensor)
        if is_tensor:
            device = joints_2d.device
            joints_2d = joints_2d.detach().cpu().numpy()

        # (N, 2) → (N, 3) ray
        rays = fisheye_camera.camera2world(joints_2d)  # returns ndarray (N, 3)

        # 보정 카메라 기준 투영: x/z, y/z
        z = 1.0 / np.tan(np.deg2rad(fov_deg / 2))  # pinhole 보정 카메라의 z값
        xy_norm = rays[:, :2] / (rays[:, 2:3] + 1e-6)  # avoid zero-division
        xy_proj = xy_norm * z

        # [-1, 1] → [0, W-1] 또는 [0, H-1]
        x_pixel = ((xy_proj[:, 0] / z) + 1) * (undist_w - 1) / 2
        y_pixel = ((xy_proj[:, 1] / z) + 1) * (undist_h - 1) / 2
        undist_joints = np.stack([x_pixel, y_pixel], axis=-1)

        if is_tensor:
            undist_joints = torch.tensor(undist_joints, dtype=torch.float32, device=device)

        return undist_joints
    
    def visualize_joint_on_images(self, undistort_img, undistorted_joint_img=None, valid=None, joint_radius=4):
        

        # RGB 변환 및 dtype 보정
        # RGB → BGR 변환 (OpenCV는 BGR 사용)
        img = undistort_img.copy()
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if undistorted_joint_img is not None:
            for i, (x, y) in enumerate(undistorted_joint_img):
                if valid is not None and valid[i] == 0:
                    continue
                if not np.isfinite(x) or not np.isfinite(y):
                    continue

                # joint 점
                cv2.circle(img, (int(round(x)), int(round(y))), joint_radius, (0, 0, 255), -1)  # 빨간색

                # joint 번호
                cv2.putText(img, str(i), (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)  # 노란색 번호

       
        cv2.imshow('Undistorted Image + Joint', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def build_forward_map(self,mapped_pixels):
        """
        mapped_pixels: (H, W, 2) — dst(i,j) 위치에서 원본 좌표를 가져오는 역방향 맵
        반환: KDTree와 보정 이미지 좌표 (dst_x, dst_y)
        """
        H, W = mapped_pixels.shape[:2]
        src_points = mapped_pixels.reshape(-1, 2)  # (H*W, 2): 원본 좌표계 기준
        dst_coords = np.stack(np.meshgrid(np.arange(W), np.arange(H)), axis=-1).reshape(-1, 2)  # (H*W, 2)

        # 유효한 원본 좌표만 사용
        valid_mask = np.all((src_points >= 0) & (src_points < np.array([W, H])), axis=1)
        src_points = src_points[valid_mask]
        dst_coords = dst_coords[valid_mask]

        tree = cKDTree(src_points)
        return tree, dst_coords

 

    def __getitem__(self, idx):
        try:
            data = copy.deepcopy(self.datalist[idx])
        except Exception as e:
            print(f'[{self.__class__.__name__}] Error loading data {idx}')
            print(e)
            exit(0)
        
        

        img_path, img_shape, bbox = data['img_path'], data['img_shape'], data['bbox']

        # img
        img_ori = load_img(img_path)
     
    
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img_ori, bbox, self.data_split)
       

        # ###Egocentric image의 입력
        crop_left = 128
        crop_right = 128
        img_ori = img_ori[:, crop_left:img_ori.shape[1] - crop_right, :]
        img_ori = self.transform(img_ori.astype(np.float32)) / 255.


        if self.data_split == 'train':
            
            # h36m gt
            joint_cam = data['joint_cam']
            if joint_cam is not None:
                dummy_cord = False
                joint_cam = joint_cam - joint_cam[self.joint_set['root_joint_idx'], None, :]  # root-relative
            else:
                # dummy cord as joint_cam
                dummy_cord = True
                joint_cam = np.zeros((self.joint_set['joint_num'], 3), dtype=np.float32)

     

            joint_img = data['joint_img']

      

            #시각화 
            # self.visualize_joint_on_images(img_ori, joint_img[:,:2]) 
            # joint_img = np.concatenate((joint_img[:, :2], joint_cam[:, 2:]), 1)  # x, y, depth

            


            if not dummy_cord: 
                joint_img[:, 2] = (joint_img[:, 2] / (cfg.body_3d_size / 2) + 1) / 2. * cfg.output_hm_shape[0]  # discretize depth
            
            joint_img_aug, joint_cam_wo_ra, joint_cam_ra, joint_valid, joint_trunc = process_db_coord(
                joint_img, joint_cam, data['joint_valid'], do_flip, img_shape,
                self.joint_set['flip_pairs'], img2bb_trans, rot, self.joint_set['joints_name'], smpl_x.joints_name)

            joint_img[..., 0] = joint_img[..., 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
            joint_img[..., 1] = joint_img[..., 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]

            
            

            # smplx coordinates and parameters
            smplx_param = data['smplx_param']
        
            smplx_joint_img, smplx_joint_cam, smplx_joint_trunc, smplx_pose, smplx_shape, smplx_expr, \
            smplx_pose_valid, smplx_joint_valid, smplx_expr_valid, smplx_mesh_cam_orig, lhand_pose, rhand_pose = process_human_model_output(
                smplx_param, self.cam_param, do_flip, img_shape, img2bb_trans, rot, 'smplx',
                joint_img=None if self.cam_param else joint_img,  # if cam not provided, we take joint_img as smplx joint 2d, which is commonly the case for             
            )

            smplx_joint_img = np.concatenate((smplx_joint_img[:, :2], smplx_joint_cam[:, 2:]), 1)
            smplx_joint_img[:, 2] = (smplx_joint_img[:, 2] / (cfg.body_3d_size / 2) + 1) / 2. * cfg.output_hm_shape[0]

            # TODO temp fix keypoints3d for renbody
            if 'RenBody' in self.__class__.__name__:
                joint_cam_ra = smplx_joint_cam.copy()
                joint_cam_wo_ra = smplx_joint_cam.copy()
                joint_cam_wo_ra[smpl_x.joint_part['lhand'], :] = joint_cam_wo_ra[smpl_x.joint_part['lhand'], :] \
                                                                + joint_cam_wo_ra[smpl_x.lwrist_idx, None, :]  # left hand root-relative
                joint_cam_wo_ra[smpl_x.joint_part['rhand'], :] = joint_cam_wo_ra[smpl_x.joint_part['rhand'], :] \
                                                                + joint_cam_wo_ra[smpl_x.rwrist_idx, None, :]  # right hand root-relative
                joint_cam_wo_ra[smpl_x.joint_part['face'], :] = joint_cam_wo_ra[smpl_x.joint_part['face'], :] \
                                                                + joint_cam_wo_ra[smpl_x.neck_idx, None,: ]  # face root-relative

            # change smplx_shape if use_betas_neutral
            # processing follows that in process_human_model_output
            if self.use_betas_neutral:
                smplx_shape = smplx_param['betas_neutral'].reshape(1, -1)
                smplx_shape[(np.abs(smplx_shape) > 3).any(axis=1)] = 0.
                smplx_shape = smplx_shape.reshape(-1)
                
            # SMPLX pose parameter validity
            # for name in ('L_Ankle', 'R_Ankle', 'L_Wrist', 'R_Wrist'):
            #     smplx_pose_valid[smpl_x.orig_joints_name.index(name)] = 0
            smplx_pose_valid = np.tile(smplx_pose_valid[:, None], (1, 3)).reshape(-1)
            # SMPLX joint coordinate validity
            # for name in ('L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel'):
            #     smplx_joint_valid[smpl_x.joints_name.index(name)] = 0
            smplx_joint_valid = smplx_joint_valid[:, None]
            smplx_joint_trunc = smplx_joint_valid * smplx_joint_trunc
            if not (smplx_shape == 0).all():
                smplx_shape_valid = True
            else: 
                smplx_shape_valid = False

            # hand and face bbox transform
            lhand_bbox, lhand_bbox_valid = self.process_hand_face_bbox(data['lhand_bbox'], do_flip, img_shape, img2bb_trans)
            rhand_bbox, rhand_bbox_valid = self.process_hand_face_bbox(data['rhand_bbox'], do_flip, img_shape, img2bb_trans)
            face_bbox, face_bbox_valid = self.process_hand_face_bbox(data['face_bbox'], do_flip, img_shape, img2bb_trans)
            if do_flip:
                lhand_bbox, rhand_bbox = rhand_bbox, lhand_bbox
                lhand_bbox_valid, rhand_bbox_valid = rhand_bbox_valid, lhand_bbox_valid
            lhand_bbox_center = (lhand_bbox[0] + lhand_bbox[1]) / 2.
            rhand_bbox_center = (rhand_bbox[0] + rhand_bbox[1]) / 2.
            face_bbox_center = (face_bbox[0] + face_bbox[1]) / 2.
            lhand_bbox_size = lhand_bbox[1] - lhand_bbox[0]
            rhand_bbox_size = rhand_bbox[1] - rhand_bbox[0]
            face_bbox_size = face_bbox[1] - face_bbox[0]

            inputs = { 'img_ori': img_ori}
            targets = {'joint_img': joint_img, # keypoints2d
                       'smplx_joint_img': smplx_joint_img, #smplx_joint_img, # projected smplx if valid cam_param, else same as keypoints2d
                       'joint_cam': joint_cam_wo_ra, # joint_cam actually not used in any loss, # raw kps3d probably without ra
                       'smplx_joint_cam': smplx_joint_cam if dummy_cord else joint_cam_ra, # kps3d with body, face, hand ra
                       'smplx_pose': smplx_pose,
                       'smplx_lhand_pose': lhand_pose,
                       'smplx_rhand_pose': rhand_pose,
                       'smplx_shape': smplx_shape,
                       'smplx_expr': smplx_expr,
                       'lhand_bbox_center': lhand_bbox_center, 'lhand_bbox_size': lhand_bbox_size,
                       'rhand_bbox_center': rhand_bbox_center, 'rhand_bbox_size': rhand_bbox_size,
                       'face_bbox_center': face_bbox_center, 'face_bbox_size': face_bbox_size}
            meta_info = {'joint_valid': joint_valid,
                         'joint_trunc': joint_trunc,
                         'joint_cam' : joint_cam,
                         'smplx_joint_valid': smplx_joint_valid if dummy_cord else joint_valid,
                         'smplx_joint_trunc': smplx_joint_trunc if dummy_cord else joint_trunc,
                         'smplx_pose_valid': smplx_pose_valid,
                         'smplx_shape_valid': float(smplx_shape_valid),
                         'smplx_expr_valid': float(smplx_expr_valid),
                         'is_3D': float(False) if dummy_cord else float(True), 
                         'lhand_bbox_valid': lhand_bbox_valid,
                         'rhand_bbox_valid': rhand_bbox_valid, 'face_bbox_valid': face_bbox_valid}
            if 'id_idx' in data:
                meta_info['id_idx'] = int(data['id_idx'])
                meta_info['id_str'] = data.get('id_str', '')
            
            
            if self.__class__.__name__  == 'SHAPY':
                meta_info['img_path'] = img_path
            
            return inputs, targets, meta_info

        # TODO: temp solution, should be more general. But SHAPY is very special
        elif self.__class__.__name__  == 'SHAPY':
            inputs = {'img': img}
            if cfg.shapy_eval_split == 'val':
                targets = {'smplx_shape': smplx_shape}
            else:
                targets = {}
            meta_info = {'img_path': img_path}
            return inputs, targets, meta_info

        else:

            joint_cam = data['joint_cam']
            if joint_cam is not None:
                dummy_cord = False
                joint_cam = joint_cam - joint_cam[self.joint_set['root_joint_idx'], None, :]  # root-relative
            else:
                # dummy cord as joint_cam
                dummy_cord = True
                joint_cam = np.zeros((self.joint_set['joint_num'], 3), dtype=np.float32)

            joint_img = data['joint_img']    ####

            joint_img = np.concatenate((joint_img[:, :2], joint_cam[:, 2:]), 1)  # x, y, depth
            if not dummy_cord:
                joint_img[:, 2] = (joint_img[:, 2] / (cfg.body_3d_size / 2) + 1) / 2. * cfg.output_hm_shape[0]  # discretize depth

            # joint_img, joint_cam, joint_cam_ra, joint_valid, joint_trunc = process_db_coord(
            #     joint_img, joint_cam, data['joint_valid'], do_flip, img_shape,
            #     self.joint_set['flip_pairs'], img2bb_trans, rot, self.joint_set['joints_name'], smpl_x.joints_name)

            smplx_param = data['smplx_param']

            smplx_joint_img, smplx_joint_cam, smplx_joint_trunc, smplx_pose, smplx_shape, smplx_expr, \
            smplx_pose_valid, smplx_joint_valid, smplx_expr_valid, smplx_mesh_cam_orig, lhand_pose, rhand_pose = process_human_model_output(
                smplx_param, self.cam_param, do_flip, img_shape, img2bb_trans, rot, 'smplx',
                joint_img=None if self.cam_param else joint_img,  # if cam not provided, we take joint_img as smplx joint 2d, which is commonly the case for             
            )

            smplx_joint_img = np.concatenate((smplx_joint_img[:, :2], smplx_joint_cam[:, 2:]), 1)
            smplx_joint_img[:, 2] = (smplx_joint_img[:, 2] / (cfg.body_3d_size / 2) + 1) / 2. * cfg.output_hm_shape[0]

            # smplx coordinates and parameters
            # smplx_param = data['smplx_param']
            # smplx_cam_trans = np.array(smplx_param['trans']) if 'trans' in smplx_param else None
            # smplx_joint_img, smplx_joint_cam, smplx_joint_trunc, smplx_pose, smplx_shape, smplx_expr, \
            # smplx_pose_valid, smplx_joint_valid, smplx_expr_valid, smplx_mesh_cam_orig = process_human_model_output(
            #     smplx_param, self.cam_param, do_flip, img_shape, img2bb_trans, rot, 'smplx',
            #     joint_img=None if self.cam_param else joint_img
            #     )  # if cam not provided, we take joint_img as smplx joint 2d, which is commonly the case for our processed humandata ####

            # inputs = {'img': img,
            #            'img_ori': img_ori}
            # targets = {}
            # # targets = {'smplx_joint_img': smplx_joint_img, ####
            #           # 'smplx_pose': smplx_pose,
            #           # 'smplx_shape': smplx_shape,
            #           #  'smplx_expr': smplx_expr,
            #           #  'smplx_cam_trans': smplx_cam_trans,
            #           # } ####
            # meta_info = {'img_path': img_path,
            #              'joint_cam': joint_cam,
            #              'bb2img_trans': bb2img_trans, ####
            # #             'gt_smplx_transl':smplx_cam_trans ####
            #             }

            inputs = { 'img_ori': img_ori}
            targets = {'joint_img': joint_img, # keypoints2d
                       'smplx_joint_img': smplx_joint_img, #smplx_joint_img, # projected smplx if valid cam_param, else same as keypoints2d
                    #    'joint_cam': joint_cam, # joint_cam actually not used in any loss, # raw kps3d probably without ra
                       'smplx_joint_cam': smplx_joint_cam,  # kps3d with body, face, hand ra
                       'smplx_pose': smplx_pose,
                       'smplx_lhand_pose': lhand_pose,
                       'smplx_rhand_pose': rhand_pose,
                       'smplx_shape': smplx_shape,
                       'smplx_expr': smplx_expr,
                       'smplx_mesh_cam': smplx_mesh_cam_orig,
                    #    'lhand_bbox_center': lhand_bbox_center, 'lhand_bbox_size': lhand_bbox_size,
                    #    'rhand_bbox_center': rhand_bbox_center, 'rhand_bbox_size': rhand_bbox_size,
                    #    'face_bbox_center': face_bbox_center, 'face_bbox_size': face_bbox_size}
                    }
            meta_info = {
                'joint_cam': joint_cam,
                'img_path': img_path
            }

            return inputs, targets, meta_info

    def process_hand_face_bbox(self, bbox, do_flip, img_shape, img2bb_trans):
        if bbox is None:
            bbox = np.array([0, 0, 1, 1], dtype=np.float32).reshape(2, 2)  # dummy value
            bbox_valid = float(False)  # dummy value
        else:
            # reshape to top-left (x,y) and bottom-right (x,y)
            bbox = bbox.reshape(2, 2)

            # flip augmentation
            if do_flip:
                bbox[:, 0] = img_shape[1] - bbox[:, 0] - 1
                bbox[0, 0], bbox[1, 0] = bbox[1, 0].copy(), bbox[0, 0].copy()  # xmin <-> xmax swap

            # make four points of the bbox
            bbox = bbox.reshape(4).tolist()
            xmin, ymin, xmax, ymax = bbox
            bbox = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float32).reshape(4, 2)

            # affine transformation (crop, rotation, scale)
            bbox_xy1 = np.concatenate((bbox, np.ones_like(bbox[:, :1])), 1)
            bbox = np.dot(img2bb_trans, bbox_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
            bbox[:, 0] = bbox[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
            bbox[:, 1] = bbox[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]

            # make box a rectangle without rotation
            xmin = np.min(bbox[:, 0])
            xmax = np.max(bbox[:, 0])
            ymin = np.min(bbox[:, 1])
            ymax = np.max(bbox[:, 1])
            bbox = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)

            bbox_valid = float(True)
            bbox = bbox.reshape(2, 2)

        return bbox, bbox_valid
    
    def visualize_joint_comparison(self, joint_gt_body, joint_out_body, mo2cap2_chain):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # GT (빨간색)
        ax.scatter(joint_gt_body[:, 0], joint_gt_body[:, 1], joint_gt_body[:, 2], c='r', label='GT', s=50)
        for connection in mo2cap2_chain:
            pts = joint_gt_body[connection]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], c='r')

        # Out (파란색)
        ax.scatter(joint_out_body[:, 0], joint_out_body[:, 1], joint_out_body[:, 2], c='b', label='Pred', s=50)
        for connection in mo2cap2_chain:
            pts = joint_out_body[connection]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], c='b')

        # joint index label for GT
        for i, (x, y, z) in enumerate(joint_gt_body):
            ax.text(x, y, z, f'{i}', fontsize=10, color='red')

        # joint index label for Pred
        for i, (x, y, z) in enumerate(joint_out_body):
            ax.text(x, y, z, f'{i}', fontsize=10, color='blue')

        ax.set_title("Joint GT vs Pred Comparison (Index)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        ax.view_init(elev=20, azim=-60)
        plt.show()

    def align_points(points_out, points_gt, pelvis_out, pelvis_gt):
        """
        points_out, points_gt: (N,3)
        pelvis_out, pelvis_gt: (3,)
        return: points_out translated so that pelvis_out == pelvis_gt
        """
        return points_out - pelvis_out[None, :] + pelvis_gt[None, :]


    def evaluate(self, outs, cur_sample_idx=None):


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        smplx_model = SMPLX(
            model_path='/home/cv1/works/SMPLer-X/common/utils/human_model_files/smplx',  # 🔹 SMPLer-X에서 사용하는 모델 경로로 변경
            batch_size=1,
            create_expression=True,
            use_pca=False
        ).to(device)


        sample_num = len(outs)
        eval_result = {
            'pa_mpvpe_all': [], 'pa_mpvpe_l_hand': [], 'pa_mpvpe_r_hand': [], 'pa_mpvpe_hand': [], 'pa_mpvpe_face': [], 'pa_mpvpe_body': [],
            'mpvpe_all': [], 'mpvpe_l_hand': [], 'mpvpe_r_hand': [], 'mpvpe_hand': [], 'mpvpe_face': [], 'mpvpe_body': [],
            'pa_mpjpe_body': [], 'pa_mpjpe_l_hand': [], 'pa_mpjpe_r_hand': [], 'pa_mpjpe_hand': [],
            'pa_mpjpe_right_arm': [], 'pa_mpjpe_left_arm': [], 'pa_mpjpe_right_leg': [], 'pa_mpjpe_left_leg': [], 'pa_mpjpe_all': [],
            # 추가: 손 포함 body+hands
            'pa_mpjpe_body_hands': [], 'mpjpe_body': [], 'mpjpe_l_hand': [], 'mpjpe_r_hand': [], 'mpjpe_hand': [], 'mpjpe_all': []
            # 이미 쓰는 포지션넷 비교가 있으면 유지
            # 'pa_mpjpe_body_pos': [],
        }

        eval_result.update({
            # 손만 (15개/hand, 1~3만 사용)
            'pa_mpjpe_l_hand_15': [],
            'pa_mpjpe_r_hand_15': [],
            'pa_mpjpe_hand_30': [],

            # 손 포함 (body 15 + hands 30)  ※ body는 네가 이미 쓰는 mo2cap2 15개 기준
            'pa_mpjpe_body_hands_45': [],

            # (선택) tip(4) 외삽해서 20개/hand로 계산
            'pa_mpjpe_l_hand_20_extrap': [],
            'pa_mpjpe_r_hand_20_extrap': [],
            'pa_mpjpe_hand_40_extrap': [],
            'pa_mpjpe_body_hands_55_extrap': [],
        })

        if getattr(cfg, 'vis', False):
            os.makedirs(cfg.vis_dir, exist_ok=True)

            import csv
            csv_file = f'{cfg.vis_dir}/{cfg.testset}_smplx_error.csv'
            file = open(csv_file, 'a', newline='')
            writer = csv.writer(file)

        for n in range(sample_num):
            out = outs[n]
            mesh_gt = out['smplx_mesh_cam_target']
            # set_trace()
            mesh_out = out['smplx_mesh_cam']


            # MPVPE from all vertices
            mesh_out_align = mesh_out - np.dot(smpl_x.J_regressor, mesh_out)[smpl_x.J_regressor_idx['pelvis'], None,
                                        :] + np.dot(smpl_x.J_regressor, mesh_gt)[smpl_x.J_regressor_idx['pelvis'], None,
                                             :]
            mpvpe_all = np.sqrt(np.sum((mesh_out_align - mesh_gt) ** 2, 1)).mean() * 1000
            eval_result['mpvpe_all'].append(mpvpe_all)
            mesh_out_align = rigid_align(mesh_out, mesh_gt)
            pa_mpvpe_all = np.sqrt(np.sum((mesh_out_align - mesh_gt) ** 2, 1)).mean() * 1000
            eval_result['pa_mpvpe_all'].append(pa_mpvpe_all)

            # MPVPE from hand vertices
            mesh_gt_lhand = mesh_gt[smpl_x.hand_vertex_idx['left_hand'], :]
            mesh_out_lhand = mesh_out[smpl_x.hand_vertex_idx['left_hand'], :]
            mesh_gt_rhand = mesh_gt[smpl_x.hand_vertex_idx['right_hand'], :]
            mesh_out_rhand = mesh_out[smpl_x.hand_vertex_idx['right_hand'], :]
            mesh_out_lhand_align = mesh_out_lhand - np.dot(smpl_x.J_regressor, mesh_out)[
                                                    smpl_x.J_regressor_idx['lwrist'], None, :] + np.dot(
                smpl_x.J_regressor, mesh_gt)[smpl_x.J_regressor_idx['lwrist'], None, :]
            mesh_out_rhand_align = mesh_out_rhand - np.dot(smpl_x.J_regressor, mesh_out)[
                                                    smpl_x.J_regressor_idx['rwrist'], None, :] + np.dot(
                smpl_x.J_regressor, mesh_gt)[smpl_x.J_regressor_idx['rwrist'], None, :]
            eval_result['mpvpe_l_hand'].append(np.sqrt(
                np.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, 1)).mean() * 1000)
            eval_result['mpvpe_r_hand'].append(np.sqrt(
                np.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, 1)).mean() * 1000)
            eval_result['mpvpe_hand'].append((np.sqrt(
                np.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, 1)).mean() * 1000 + np.sqrt(
                np.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, 1)).mean() * 1000) / 2.)
            mesh_out_lhand_align = rigid_align(mesh_out_lhand, mesh_gt_lhand)
            mesh_out_rhand_align = rigid_align(mesh_out_rhand, mesh_gt_rhand)
            eval_result['pa_mpvpe_l_hand'].append(np.sqrt(
                np.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, 1)).mean() * 1000)
            eval_result['pa_mpvpe_r_hand'].append(np.sqrt(
                np.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, 1)).mean() * 1000)
            eval_result['pa_mpvpe_hand'].append((np.sqrt(
                np.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, 1)).mean() * 1000 + np.sqrt(
                np.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, 1)).mean() * 1000) / 2.)

            # # MPVPE from face vertices
            mesh_gt_face = mesh_gt[smpl_x.face_vertex_idx, :]
            mesh_out_face = mesh_out[smpl_x.face_vertex_idx, :]
            mesh_out_face_align = mesh_out_face - np.dot(smpl_x.J_regressor, mesh_out)[smpl_x.J_regressor_idx['neck'],
                                                  None, :] + np.dot(smpl_x.J_regressor, mesh_gt)[
                                                             smpl_x.J_regressor_idx['neck'], None, :]
            eval_result['mpvpe_face'].append(
                np.sqrt(np.sum((mesh_out_face_align - mesh_gt_face) ** 2, 1)).mean() * 1000)
            mesh_out_face_align = rigid_align(mesh_out_face, mesh_gt_face)
            eval_result['pa_mpvpe_face'].append(
                np.sqrt(np.sum((mesh_out_face_align - mesh_gt_face) ** 2, 1)).mean() * 1000)

            
            # set_trace()

            # MPJPE from body joints

            mesh_out_pelvis_align = mesh_out - np.dot(smpl_x.J_regressor, mesh_out)[smpl_x.J_regressor_idx['pelvis'], None,
                                        :] + np.dot(smpl_x.J_regressor, mesh_gt)[smpl_x.J_regressor_idx['pelvis'], None,
                                             :]
            joint_gt_all = np.dot(smpl_x.J_regressor, mesh_gt)
            joint_out_all = np.dot(smpl_x.J_regressor, mesh_out_pelvis_align)
            # joint_out_all = joint_out_all - np.dot(smpl_x.J_regressor, mesh_out)[smpl_x.J_regressor_idx['pelvis'], None,
            #                             :] + np.dot(smpl_x.J_regressor, mesh_gt)[smpl_x.J_regressor_idx['pelvis'], None,
            #                                  :]
            
            eval_result['mpjpe_body'].append(
                np.sqrt(np.sum((joint_out_all - joint_gt_all) ** 2, 1)).mean() * 1000)
            joint_out_all_align = rigid_align(joint_out_all, joint_gt_all)
            pa_mpjpe_body = np.sqrt(np.sum((joint_out_all_align - joint_gt_all) ** 2, 1)).mean() * 1000
            eval_result['pa_mpjpe_body'].append(pa_mpjpe_body)

            # MPJPE from hand joints
            joint_gt_lhand = np.dot(smpl_x.orig_hand_regressor['left'], mesh_gt)
            joint_out_lhand = np.dot(smpl_x.orig_hand_regressor['left'], mesh_out)
            joint_out_lhand = joint_out_lhand - np.dot(smpl_x.J_regressor, mesh_out)[smpl_x.J_regressor_idx['lwrist'], None,
                                        :] + np.dot(smpl_x.J_regressor, mesh_gt)[smpl_x.J_regressor_idx['lwrist'], None,
                                             :]
            joint_out_lhand_align = rigid_align(joint_out_lhand, joint_gt_lhand)
            joint_gt_rhand = np.dot(smpl_x.orig_hand_regressor['right'], mesh_gt)
            joint_out_rhand = np.dot(smpl_x.orig_hand_regressor['right'], mesh_out)
            joint_out_rhand = joint_out_rhand - np.dot(smpl_x.J_regressor, mesh_out)[smpl_x.J_regressor_idx['rwrist'], None,
                                        :] + np.dot(smpl_x.J_regressor, mesh_gt)[smpl_x.J_regressor_idx['rwrist'], None,
                                             :]
            joint_out_rhand_align = rigid_align(joint_out_rhand, joint_gt_rhand)
            eval_result['mpjpe_l_hand'].append(np.sqrt(
                np.sum((joint_out_lhand - joint_gt_lhand) ** 2, 1)).mean() * 1000)
            eval_result['mpjpe_r_hand'].append(np.sqrt(
                np.sum((joint_out_rhand - joint_gt_rhand) ** 2, 1)).mean() * 1000)
            eval_result['mpjpe_hand'].append((np.sqrt(
                np.sum((joint_out_lhand - joint_gt_lhand) ** 2, 1)).mean() * 1000 + np.sqrt(
                np.sum((joint_out_rhand - joint_gt_rhand) ** 2, 1)).mean() * 1000) / 2.)
            eval_result['pa_mpjpe_l_hand'].append(np.sqrt(
                np.sum((joint_out_lhand_align - joint_gt_lhand) ** 2, 1)).mean() * 1000)
            eval_result['pa_mpjpe_r_hand'].append(np.sqrt(
                np.sum((joint_out_rhand_align - joint_gt_rhand) ** 2, 1)).mean() * 1000)
            eval_result['pa_mpjpe_hand'].append((np.sqrt(
                np.sum((joint_out_lhand_align - joint_gt_lhand) ** 2, 1)).mean() * 1000 + np.sqrt(
                np.sum((joint_out_rhand_align - joint_gt_rhand) ** 2, 1)).mean() * 1000) / 2.)


            # === (evaluate 내부) 2D 시각화 옵션 ===
            VIS_2D = getattr(cfg, "vis_2d_pose", False)
            VIS_2D_ROOT = os.path.join(cfg.vis_dir, "pose2d_overlay")

            if VIS_2D:
                print("[VIS2D] entered VIS_2D block, n=", n)
                # set_trace()
                # 0) img path
                img_path = out['img_path']

                # (중요) img_path가 상대경로일 수 있으니 보정
                if isinstance(img_path, str) and (not os.path.exists(img_path)):
                    # self.img_dir가 있으면 붙여보기
                    if getattr(self, "img_dir", None):
                        cand = os.path.join(self.img_dir, img_path)
                        if os.path.exists(cand):
                            img_path = cand

                if not (isinstance(img_path, str) and os.path.exists(img_path)):
                    # 최초 1회만 원인 출력
                    if n == 0:
                        print("[VIS2D] img_path invalid:", img_path)
                        print("[VIS2D] out keys:", list(out.keys())[:50], "...")
                    # skip
                    pass
                else:
                    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if img_bgr is None:
                        if n == 0:
                            print("[VIS2D] cv2.imread failed:", img_path)
                        pass
                    else:
                        H_img, W_img = img_bgr.shape[:2]

                        # 1) GT / Pred 2D (fallback 키들)
                        gt2d = out["smplx_joint_img"][:, :2]
                        pred2d = out["smplx_joint_proj"]

                        if gt2d is None or pred2d is None:
                            if n == 0:
                                print("[VIS2D] missing 2D keys. gt2d:", gt2d is not None, "pred2d:", pred2d is not None)
                                print("[VIS2D] out keys:", [k for k in out.keys() if "smplx" in k and "joint" in k])
                            pass
                        

                        hm_h = 16
                        hm_w = 12

                        gt_xy = maybe_scale_from_hm_to_img(gt2d, W_img, H_img, hm_w, hm_h)
                        pr_xy = maybe_scale_from_hm_to_img(pred2d, W_img, H_img, hm_w, hm_h)

                        # bb2img_trans = out.get("bb2img_trans", None)
                        # if bb2img_trans is not None:
                        #     bb2img_trans = np.asarray(bb2img_trans, dtype=np.float32).reshape(3, 3)
                        #     gt_xy = _apply_homography_xy(gt_xy, bb2img_trans)
                        #     pr_xy = _apply_homography_xy(pr_xy, bb2img_trans)

                        valid_gt = None
                        valid_pr = None

                        # body(0~24)에서 ear/eye/nose 제거
                        DROP_BODY_FACE = {20, 21, 22, 23, 24}

                        body_wo_face = [i for i in range(25) if i not in DROP_BODY_FACE]  # 0~24 중 얼굴 제외
                        hands = list(range(25, 65))  # 25~64 (양손)

                        subset_idx = np.array(body_wo_face + hands, dtype=np.int32)

                        # subset_idx = np.arange(BODY_HANDS_NUM)  # 0~64
                        overlay = draw_2d_pose_overlay(
                            img_bgr,
                            joints_gt=gt_xy,
                            joints_pred=pr_xy,
                            valid_gt=valid_gt,
                            valid_pred=valid_pr,
                            edges=BODY_HANDS_EDGES,
                            subset_idx=subset_idx,
                            gt_color=(0, 255, 0),
                            pred_color=(0, 0, 255),
                            radius=3,
                            thickness=2,
                            draw_index=False
                        )

                        VIS_2D_ROOT = os.path.join(cfg.vis_dir, "pose2d_overlay")

                        subject_scene_dir = self._get_subject_scene_dir(img_path)
                        if subject_scene_dir:
                            save_dir = os.path.join(VIS_2D_ROOT, subject_scene_dir)
                        else:
                            save_dir = VIS_2D_ROOT

                        os.makedirs(save_dir, exist_ok=True)

                        base = os.path.splitext(os.path.basename(img_path))[0]
                        vis_idx = self.save_idx  # npz랑 동일 idx 쓰고 싶으면

                        save_path = os.path.join(save_dir, f"{base}_{vis_idx:06d}.jpg")
                        ok = cv2.imwrite(save_path, overlay)
                        if not ok:
                            print("[VIS2D] write failed:", save_path)

                        if getattr(cfg, "vis_2d_pose_show", False):
                            cv2.imshow("2D Pose Overlay (Body+Hands) GT=Green Pred=Red", overlay)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()




            if getattr(cfg, 'vis', False):
                img_path = out['img_path']
                base = os.path.splitext(os.path.basename(img_path))[0] if isinstance(img_path, str) else f"{self.save_idx:06d}"
                if hasattr(self, 'img_dir') and self.img_dir and isinstance(img_path, str):
                    rel_img_path = os.path.relpath(img_path, start=self.img_dir)
                elif isinstance(img_path, str):
                    rel_img_path = os.path.basename(img_path)
                else:
                    rel_img_path = base

                # 데이터셋 경로에서 {subject}/{scene} 부분만 추출
                subject_scene_dir = self._get_subject_scene_dir(img_path)

                if subject_scene_dir:
                    npz_dir = os.path.join(cfg.vis_dir, subject_scene_dir)
                else:
                    npz_dir = cfg.vis_dir

                os.makedirs(npz_dir, exist_ok=True)

                smplx_pred = {}
                smplx_pred['global_orient'] = out['smplx_root_pose'].reshape(-1,3)
                smplx_pred['body_pose'] = out['smplx_body_pose'].reshape(-1,3)
                smplx_pred['left_hand_pose'] = out['smplx_lhand_pose'].reshape(-1,3)
                smplx_pred['right_hand_pose'] = out['smplx_rhand_pose'].reshape(-1,3)
                smplx_pred['jaw_pose'] = out['smplx_jaw_pose'].reshape(-1,3)
                smplx_pred['leye_pose'] = np.zeros((1, 3))
                smplx_pred['reye_pose'] = np.zeros((1, 3))
                smplx_pred['betas'] = out['smplx_shape'].reshape(-1,10)
                smplx_pred['expression'] = out['smplx_expr'].reshape(-1,10)
                # smplx_pred['transl'] =  out['gt_smplx_transl'].reshape(-1,3)
                smplx_pred['img_path'] = rel_img_path

                npz_path = os.path.join(npz_dir, f'{base}_{self.save_idx:06d}.npz')
                np.savez(npz_path, **smplx_pred)
                try:
                    writer.writerow([self.save_idx, rel_img_path, npz_path])
                except Exception:
                    pass

                # save img path and error
                # 위에서 계산된 값들 중 hand pa mpvpe
                pa_mpvpe_hand_cur = eval_result['pa_mpvpe_hand'][-1]   # 방금 append된 값
                pa_mpvpe_lhand_cur = eval_result['pa_mpvpe_l_hand'][-1]
                pa_mpvpe_rhand_cur = eval_result['pa_mpvpe_r_hand'][-1]

                new_line = [
                    self.save_idx,
                    rel_img_path,
                    pa_mpjpe_body,
                    pa_mpvpe_hand_cur,
                    pa_mpvpe_lhand_cur,
                    pa_mpvpe_rhand_cur,
                ]
                writer.writerow(new_line)

                self.save_idx += 1

        if getattr(cfg, 'vis', False):
            file.close()

        return eval_result
            
    def print_eval_result(self, eval_result):
        print(f'======{cfg.testset}======')
        print(f'{cfg.vis_dir}')
        print('PA MPVPE (All): %.2f mm' % np.mean(eval_result['pa_mpvpe_all']))
        print('PA MPVPE (L-Hands): %.2f mm' % np.mean(eval_result['pa_mpvpe_l_hand']))
        print('PA MPVPE (R-Hands): %.2f mm' % np.mean(eval_result['pa_mpvpe_r_hand']))
        print('PA MPVPE (Hands): %.2f mm' % np.mean(eval_result['pa_mpvpe_hand']))
        # print('PA MPVPE (Face): %.2f mm' % np.mean(eval_result['pa_mpvpe_face']))
        print()

        print('MPVPE (All): %.2f mm' % np.mean(eval_result['mpvpe_all']))
        print('MPVPE (L-Hands): %.2f mm' % np.mean(eval_result['mpvpe_l_hand']))
        print('MPVPE (R-Hands): %.2f mm' % np.mean(eval_result['mpvpe_r_hand']))
        print('MPVPE (Hands): %.2f mm' % np.mean(eval_result['mpvpe_hand']))
        # print('MPVPE (Face): %.2f mm' % np.mean(eval_result['mpvpe_face']))
        print()

        print('MPJPE (Body): %.2f mm' % np.mean(eval_result['mpjpe_body']))

        print('PA MPJPE (Body): %.2f mm' % np.mean(eval_result['pa_mpjpe_body']))
        
        # print('PA MPJPE (PositionNet Body): %.2f mm' % np.mean(eval_result['pa_mpjpe_body_pos']))
        # print('PA MPJPE (Left Arm): %.2f mm' % np.mean(eval_result['pa_mpjpe_left_arm']))
        # print('PA MPJPE (Right Arm): %.2f mm' % np.mean(eval_result['pa_mpjpe_right_arm']))
        # print('PA MPJPE (Left Leg): %.2f mm' % np.mean(eval_result['pa_mpjpe_left_leg']))
        # print('PA MPJPE (Right Leg): %.2f mm' % np.mean(eval_result['pa_mpjpe_right_leg']))

        # print('PA MPJPE (L-Hand 15): %.2f mm' % np.mean(eval_result['pa_mpjpe_l_hand_15']))
        # print('PA MPJPE (R-Hand 15): %.2f mm' % np.mean(eval_result['pa_mpjpe_r_hand_15']))
        # print('PA MPJPE (Hands 30): %.2f mm' % np.mean(eval_result['pa_mpjpe_hand_30']))
        # print('PA MPJPE (Body+Hands 45): %.2f mm' % np.mean(eval_result['pa_mpjpe_body_hands_45']))

    
        print('MPJPE (L-Hands): %.2f mm' % np.mean(eval_result['mpjpe_l_hand']))
        print('MPJPE (R-Hands): %.2f mm' % np.mean(eval_result['mpjpe_r_hand']))
        print('MPJPE (Hands): %.2f mm' % np.mean(eval_result['mpjpe_hand']))


        print('PA MPJPE (L-Hands): %.2f mm' % np.mean(eval_result['pa_mpjpe_l_hand']))
        print('PA MPJPE (R-Hands): %.2f mm' % np.mean(eval_result['pa_mpjpe_r_hand']))
        print('PA MPJPE (Hands): %.2f mm' % np.mean(eval_result['pa_mpjpe_hand']))

        # print()
        # print(f"{np.mean(eval_result['pa_mpvpe_all'])},{np.mean(eval_result['pa_mpvpe_l_hand'])},{np.mean(eval_result['pa_mpvpe_r_hand'])},{np.mean(eval_result['pa_mpvpe_hand'])},{np.mean(eval_result['pa_mpvpe_face'])},"
        # f"{np.mean(eval_result['mpvpe_all'])},{np.mean(eval_result['mpvpe_l_hand'])},{np.mean(eval_result['mpvpe_r_hand'])},{np.mean(eval_result['mpvpe_hand'])},{np.mean(eval_result['mpvpe_face'])},"
        # f"{np.mean(eval_result['pa_mpjpe_body'])},{np.mean(eval_result['pa_mpjpe_l_hand'])},{np.mean(eval_result['pa_mpjpe_r_hand'])},{np.mean(eval_result['pa_mpjpe_hand'])}")
        # print()


        # f = open(os.path.join(cfg.result_dir, 'result.txt'), 'w')
        # f.write(f'{cfg.testset} dataset \n')
        # f.write('PA MPVPE (All): %.2f mm\n' % np.mean(eval_result['pa_mpvpe_all']))
        # f.write('PA MPVPE (L-Hands): %.2f mm' % np.mean(eval_result['pa_mpvpe_l_hand']))
        # f.write('PA MPVPE (R-Hands): %.2f mm' % np.mean(eval_result['pa_mpvpe_r_hand']))
        # f.write('PA MPVPE (Hands): %.2f mm\n' % np.mean(eval_result['pa_mpvpe_hand']))
        # f.write('PA MPVPE (Face): %.2f mm\n' % np.mean(eval_result['pa_mpvpe_face']))
        # f.write('MPVPE (All): %.2f mm\n' % np.mean(eval_result['mpvpe_all']))
        # f.write('MPVPE (L-Hands): %.2f mm' % np.mean(eval_result['mpvpe_l_hand']))
        # f.write('MPVPE (R-Hands): %.2f mm' % np.mean(eval_result['mpvpe_r_hand']))
        # f.write('MPVPE (Hands): %.2f mm' % np.mean(eval_result['mpvpe_hand']))
        # f.write('MPVPE (Face): %.2f mm\n' % np.mean(eval_result['mpvpe_face']))
        # f.write('PA MPJPE (Body): %.2f mm\n' % np.mean(eval_result['pa_mpjpe_body']))
        # f.write('PA MPJPE (L-Hands): %.2f mm' % np.mean(eval_result['pa_mpjpe_l_hand']))
        # f.write('PA MPJPE (R-Hands): %.2f mm' % np.mean(eval_result['pa_mpjpe_r_hand']))
        # f.write('PA MPJPE (Hands): %.2f mm\n' % np.mean(eval_result['pa_mpjpe_hand']))
        # f.write(f"{np.mean(eval_result['pa_mpvpe_all'])},{np.mean(eval_result['pa_mpvpe_l_hand'])},{np.mean(eval_result['pa_mpvpe_r_hand'])},{np.mean(eval_result['pa_mpvpe_hand'])},{np.mean(eval_result['pa_mpvpe_face'])},"
        # f"{np.mean(eval_result['mpvpe_all'])},{np.mean(eval_result['mpvpe_l_hand'])},{np.mean(eval_result['mpvpe_r_hand'])},{np.mean(eval_result['mpvpe_hand'])},{np.mean(eval_result['mpvpe_face'])},"
        # f"{np.mean(eval_result['pa_mpjpe_body'])},{np.mean(eval_result['pa_mpjpe_l_hand'])},{np.mean(eval_result['pa_mpjpe_r_hand'])},{np.mean(eval_result['pa_mpjpe_hand'])}")

        # if getattr(cfg, 'eval_on_train', False):
        #     import csv
        #     csv_file = f'{cfg.root_dir}/output/{cfg.testset}_eval_on_train.csv'
        #     exp_id = cfg.exp_name.split('_')[1]
        #     new_line = [exp_id,np.mean(eval_result['pa_mpvpe_all']),np.mean(eval_result['pa_mpvpe_l_hand']),np.mean(eval_result['pa_mpvpe_r_hand']),np.mean(eval_result['pa_mpvpe_hand']),np.mean(eval_result['pa_mpvpe_face']),
        #                 np.mean(eval_result['mpvpe_all']),np.mean(eval_result['mpvpe_l_hand']),np.mean(eval_result['mpvpe_r_hand']),np.mean(eval_result['mpvpe_hand']),np.mean(eval_result['mpvpe_face']),
        #                 np.mean(eval_result['pa_mpjpe_body']),np.mean(eval_result['pa_mpjpe_l_hand']),np.mean(eval_result['pa_mpjpe_r_hand']),np.mean(eval_result['pa_mpjpe_hand'])]

        #     # Append the new line to the CSV file
        #     with open(csv_file, 'a', newline='') as file:
        #         writer = csv.writer(file)
        #         writer.writerow(new_line)

    def decompress_keypoints(self, humandata) -> None:
        """If a key contains 'keypoints', and f'{key}_mask' is in self.keys(),
        invalid zeros will be inserted to the right places and f'{key}_mask'
        will be unlocked.

        Raises:
            KeyError:
                A key contains 'keypoints' has been found
                but its corresponding mask is missing.
        """
        assert bool(humandata['__keypoints_compressed__']) is True
        key_pairs = []
        for key in humandata.files:
            if key not in KPS2D_KEYS + KPS3D_KEYS:
                continue
            mask_key = f'{key}_mask'
            if mask_key in humandata.files:
                print(f'Decompress {key}...')
                key_pairs.append([key, mask_key])
        decompressed_dict = {}
        for kpt_key, mask_key in key_pairs:
            mask_array = np.asarray(humandata[mask_key])
            compressed_kpt = humandata[kpt_key]
            kpt_array = \
                self.add_zero_pad(compressed_kpt, mask_array)
            decompressed_dict[kpt_key] = kpt_array
        del humandata
        return decompressed_dict

    def add_zero_pad(self, compressed_array: np.ndarray,
                         mask_array: np.ndarray) -> np.ndarray:
        """Pad zeros to a compressed keypoints array.

        Args:
            compressed_array (np.ndarray):
                A compressed keypoints array.
            mask_array (np.ndarray):
                The mask records compression relationship.

        Returns:
            np.ndarray:
                A keypoints array in full-size.
        """
        assert mask_array.sum() == compressed_array.shape[1]
        data_len, _, dim = compressed_array.shape
        mask_len = mask_array.shape[0]
        ret_value = np.zeros(
            shape=[data_len, mask_len, dim], dtype=compressed_array.dtype)
        valid_mask_index = np.where(mask_array == 1)[0]
        ret_value[:, valid_mask_index, :] = compressed_array
        return ret_value
