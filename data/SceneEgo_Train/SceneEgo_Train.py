import os
import os.path as osp
import numpy as np
import torch
from humandata import HumanDataset
import smplx
import open3d as o3d


class SceneEgo_Train(HumanDataset):
    def __init__(self, transform, data_split):
        """
        SceneEgo_Train은 SceneEgo 데이터셋의 학습용 데이터를 로딩합니다.
        이미지: SceneEgo_train/train/{subject}/imgs/img_%06d.jpg
        주석: SceneEgo_train/train/human_data_train/humandata.npz
        """
        super(SceneEgo_Train, self).__init__(transform, data_split)

       
        self.data_dir = '/media/cv1/SeagateHDD2TB/SceneEgo_train/train/human_data_train'
        self.img_root = '/media/cv1/SeagateHDD2TB/SceneEgo_train/train'
        self.img_shape = (1024, 1280)
        self.annot_path = osp.join(self.data_dir, 'humandata.npz')
        self.use_cache = False
        self.annot_path_cache = osp.join(self.data_dir, 'cache', 'humandata_cache.npz')

        if not osp.exists(self.annot_path):
            raise FileNotFoundError(f"Annotation file not found: {self.annot_path}")

        if self.use_cache and osp.exists(self.annot_path_cache):
            print(f"[{self.__class__.__name__}] Loading cache from {self.annot_path_cache}")
            self.datalist = self.load_cache(self.annot_path_cache)
        else:
            print(f"[{self.__class__.__name__}] Loading annotations from {self.annot_path}")
            data = np.load(self.annot_path, allow_pickle=True)

            self.image_paths = data['image_path']  
            self.bbox_xywh = data['bbox_xywh']
            self.keypoints2d = data['keypoints2d_smplx']
            self.keypoints2d_mask = data['keypoints2d_smplx_mask']
            self.keypoints3d = data['keypoints3d_smplx']
            self.keypoints3d_mask = data['keypoints3d_smplx_mask']
            self.smplx_params = data['smplx']
            self.cam_param = {}


            self.datalist = [
                {
                    'img_path': osp.join(self.img_root, img_path),
                    'img_shape': self.img_shape,
                    'bbox': bbox,
                    'keypoints2d': kp2d,
                    'keypoints2d_mask': kp2d_mask,
                    'keypoints3d': kp3d,
                    'keypoints3d_mask': kp3d_mask,
                    'smplx_param': {
                        'root_pose': np.zeros((1, 3)),
                        'body_pose': smplx_dict.get('body_pose', np.zeros((69,))).reshape(23, 3)[:21, :],
                        'shape': smplx_dict.get('betas', np.zeros((10,))),
                        'trans': np.zeros((3,)),
                        'expr': smplx_dict.get('expr', np.zeros((10,))),
                        'leye_pose': smplx_dict.get('leye_pose', np.zeros((1, 3))),
                        'reye_pose': smplx_dict.get('reye_pose', np.zeros((1, 3))),
                    },
                    'smpl_param': {
                        'global_orient': np.zeros((1, 3)),
                        'body_pose': smplx_dict.get('body_pose', np.zeros((69,))).reshape(23, 3),
                        'shape': smplx_dict.get('betas', np.zeros((10,)))
                    },
                    'joint_img': kp2d,
                    'joint_valid': kp2d_mask.reshape(-1, 1),
                    'joint_cam': None,
                    'is_3D': None,
                    'lhand_bbox': None,
                    'rhand_bbox': None,
                    'face_bbox': None
                }
                for img_path, bbox, kp2d, kp2d_mask, kp3d, kp3d_mask, smplx_dict in zip(
                    self.image_paths, self.bbox_xywh, self.keypoints2d,
                    self.keypoints2d_mask, self.keypoints3d,
                    self.keypoints3d_mask, self.smplx_params
                )
            ]

            if self.use_cache:
                self.save_cache(self.annot_path_cache, self.datalist)

            print(f"[{self.__class__.__name__}] Loaded {len(self.datalist)} samples.")
