import os
import os.path as osp
import numpy as np
import torch
from humandata import HumanDataset
import smplx
import open3d as o3d
from pathlib import Path
from pdb import set_trace


class SceneEgo_Test(HumanDataset):
    def __init__(self, transform, data_split):
        super(SceneEgo_Test, self).__init__(transform, data_split)

        # 경로 설정
        self.data_dir = '/media/cv1/SeagateHDD2TB/SceneEgo_test/human_data_test_smplx'
        self.img_shape = (1024, 1280)
        self.annot_path = osp.join(self.data_dir, 'humandata.npz')
        self.annot_path_cache = osp.join(self.data_dir, 'cache', 'humandata_cache.npz')
        self.use_cache = False

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

        self.datalist = []
        
        for img_path, bbox, kp2d, kp2d_mask, kp3d, kp3d_mask, smplx_d in zip(
            self.image_paths, self.bbox_xywh, self.keypoints2d, self.keypoints2d_mask,
            self.keypoints3d, self.keypoints3d_mask, self.smplx_params
        ):

            item = {
                'img_path': img_path,
                'img_shape': self.img_shape,
                'bbox': bbox,
                'keypoints2d': kp2d,
                'keypoints2d_mask': kp2d_mask,
                'keypoints3d': kp3d,
                'keypoints3d_mask': kp3d_mask,
                'smplx_param': {
                    'root_pose': smplx_d.get('global_orient', np.zeros((3,))),
                    'body_pose': smplx_d.get('body_pose', np.zeros((3,))),
                    'shape': smplx_d.get('betas', np.zeros((3,))),
                    'trans': smplx_d.get('transl', np.zeros((3,))),
                    'expr': smplx_d.get('expr', np.zeros((10,))),
                    'lhand_pose': smplx_d.get('left_hand_pose', np.zeros((15, 3))),
                    'rhand_pose': smplx_d.get('right_hand_pose', np.zeros((15, 3))),
                    'leye_pose': smplx_d.get('leye_pose', np.zeros((1, 3))),
                    'reye_pose': smplx_d.get('reye_pose', np.zeros((1, 3))),
                    'jaw_pose': smplx_d.get('jaw_pose', np.zeros((1, 3))),
                    'lhand_valid': True,
                    'rhand_valid': True,
                    'face_valid': True
                },
                'joint_img': kp2d,
                'joint_valid': kp2d_mask.reshape(-1, 1),
                'joint_cam': None,
                'lhand_bbox': None,
                'rhand_bbox': None,
                'face_bbox': None,
                
            }
            self.datalist.append(item)

        # print(f"[{self.__class__.__name__}] Subsampled: {len(self.datalist)} / {N} (interval={sample_interval})")

        self.visualize_mesh('/home/cv1/works/SMPLer-X/common/utils/human_model_files/smplx')  ## GT 시각화하고 싶으면

        if self.use_cache:
            self.save_cache(self.annot_path_cache, self.datalist)

        # print(f"[{self.__class__.__name__}] IDs({self.num_ids}): {self.IDs}")
        print(f"[{self.__class__.__name__}] Loaded {len(self.datalist)} samples.")

    def visualize_mesh(self, smplx_model_path, sample_idx=0, gender='neutral'):
            """
            주어진 sample의 SMPL-X 파라미터로부터 mesh를 생성하고 open3d로 시각화합니다.
            """
            smplx_model = smplx.SMPLX(
                model_path=smplx_model_path,
                gender=gender,
                batch_size=1,
                use_pca=False,
                create_expression=True,
                create_jaw_pose=True,
                create_left_hand_pose=True,
                create_right_hand_pose=True,
                create_leye_pose=True,
                create_reye_pose=True,
                create_transl=True,
            ).to("cpu")

            param = self.datalist[sample_idx]['smplx_param']
            with torch.no_grad():
                smplx_output = smplx_model(
                    global_orient=torch.tensor(param['root_pose'], dtype=torch.float32).view(1, 3),
                    body_pose=torch.tensor(param['body_pose'], dtype=torch.float32).view(1, -1),
                    betas=torch.tensor(param['shape'], dtype=torch.float32).view(1, -1),  # (1,10)
                    transl=torch.tensor(param['trans'], dtype=torch.float32).view(1, 3),
                    expression=torch.tensor(param['expr'], dtype=torch.float32).view(1, -1),
                    jaw_pose=torch.tensor(param['jaw_pose'], dtype=torch.float32).view(1, 3),
                    left_hand_pose=torch.tensor(param['lhand_pose'], dtype=torch.float32).view(1, -1),
                    right_hand_pose=torch.tensor(param['rhand_pose'], dtype=torch.float32).view(1, -1),
                    leye_pose=torch.tensor(param['leye_pose'], dtype=torch.float32).view(1, 3),
                    reye_pose=torch.tensor(param['reye_pose'], dtype=torch.float32).view(1, 3),
                )
                verts = smplx_output.vertices[0].cpu().numpy()
                faces = smplx_model.faces

                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(verts)
                mesh.triangles = o3d.utility.Vector3iVector(faces)
                mesh.compute_vertex_normals()

                o3d.visualization.draw_geometries([mesh])

# visualize

# class EgoWholeBody(HumanDataset):
#     def __init__(self, transform, data_split):
#         """
#         PseudoGTDataset은 HumanDataset을 상속하며, Humandata.npz 파일을 로딩하여 학습 및 평가에 필요한 데이터를 초기화합니다.
        
#         Args:
#             transform: 이미지 전처리를 위한 transform 함수
#             data_split: 데이터셋 구분 ('train' 또는 'test')
#             data_dir: Humandata.npz가 저장된 디렉토리
#         """
#         super(EgoWholeBody, self).__init__(transform, data_split)

#         # 경로 설정
#         self.data_dir = '/media/cv1/SeagateHDD2TB/EgoWholeMocap/human_data_train'
#         self.img_shape = (1024, 1280)  # 이미지 크기 설정
#         self.annot_path = osp.join(self.data_dir, 'humandata.npz')
#         self.annot_path_cache = osp.join(self.data_dir, 'cache', 'humandata_cache.npz')
#         self.use_cache = False

#         if not osp.exists(self.annot_path):
#             raise FileNotFoundError(f"Annotation file not found: {self.annot_path}")

#         # 캐시 사용 여부에 따른 데이터 로딩
#         if self.use_cache and osp.exists(self.annot_path_cache):
#             print(f"[{self.__class__.__name__}] Loading cache from {self.annot_path_cache}")
#             self.datalist = self.load_cache(self.annot_path_cache)
#         else:
#             print(f"[{self.__class__.__name__}] Loading annotations from {self.annot_path}")
#             data = np.load(self.annot_path, allow_pickle=True)

#             # 필요한 키 데이터 로딩
#             self.image_paths = data['image_path']
#             self.bbox_xywh = data['bbox_xywh']
#             self.keypoints2d = data['keypoints2d_smplx']
#             self.keypoints2d_mask = data['keypoints2d_smplx_mask']
#             self.keypoints3d = data['keypoints3d_smplx']
#             self.keypoints3d_mask = data['keypoints3d_smplx_mask']
#             self.smplx_params = data['smplx']
#             self.cam_param = {}
            

#             # datalist 생성
#             self.datalist = [
#             {
#                 'img_path': osp.join(self.data_dir, img_path),
#                 'img_shape': self.img_shape,
#                 'bbox': bbox,
#                 'keypoints2d': kp2d,
#                 'keypoints2d_mask': kp2d_mask,
#                 'keypoints3d': kp3d,
#                 'keypoints3d_mask': kp3d_mask,
#                 'smplx_param': {
#                     'root_pose': smplx.get('global_orient', np.zeros((3,))),
#                     # 'root_pose':np.zeros((3,)),
#                     'body_pose': smplx.get('body_pose', np.zeros((63,))).reshape(21,3),
#                     'shape': smplx.get('betas', np.zeros((10,))),
#                     'trans': smplx.get('transl', np.zeros((3,))),
#                     # 'lhand_pose': smplx.get('lhand_pose', np.zeros((15, 3))),
#                     # 'rhand_pose': smplx.get('rhand_pose', np.zeros((15, 3))),
#                     'expr': smplx.get('expr', np.zeros((10,))),
#                     'leye_pose': smplx.get('leye_pose', np.zeros((1,3))),   # 왼쪽 눈 포즈
#                     'reye_pose': smplx.get('reye_pose', np.zeros((1,3))),   # 오른쪽 눈 포즈
#                     # 'jaw_pose': smplx.get('jaw_pose', np.zeros((1,3)))      # 턱 포즈
#                 },
#                 'joint_img': kp2d,
#                 'joint_valid': kp2d_mask.reshape(-1, 1),
#                 'joint_cam': kp3d,
#                 'lhand_bbox': None,
#                 'rhand_bbox': None,
#                 'face_bbox': None
#             }
#                 for img_path, bbox, kp2d, kp2d_mask, kp3d, kp3d_mask, smplx in zip(
#                     self.image_paths, self.bbox_xywh, self.keypoints2d, 
#                     self.keypoints2d_mask, self.keypoints3d, 
#                     self.keypoints3d_mask, self.smplx_params
#                 )
                
#             ]
#             # print("-----------body_pose_shape--------------", self.datalist[0]['smplx_param']['body_pose'].shape)

            
#             # self.visualize_mesh(smplx_model_path='/home/cv1/works/smplify-x/models/smplx', sample_idx=0)

#             if self.use_cache:
#                 self.save_cache(self.annot_path_cache, self.datalist)

#             print(f"[{self.__class__.__name__}] Loaded {len(self.datalist)} samples.")


#     def visualize_mesh(self, smplx_model_path, sample_idx=0, gender='neutral'):
#             """
#             주어진 sample의 SMPL-X 파라미터로부터 mesh를 생성하고 open3d로 시각화합니다.
#             """
#             smplx_model = smplx.SMPLX(
#                 model_path=smplx_model_path,
#                 gender=gender,
#                 batch_size=1,
#                 use_pca=False,
#                 create_global_orient=True,
#                 create_body_pose=True,
#                 create_betas=True,
#                 create_transl=True,
#                 create_expression=True,
#                 create_left_eye_pose=True,
#                 create_right_eye_pose=True,
#             ).to('cpu')

#             param = self.datalist[sample_idx]['smplx_param']
#             with torch.no_grad():
#                 smplx_output = smplx_model(
#                     global_orient=torch.tensor(param['root_pose'], dtype=torch.float32).unsqueeze(0),
#                     body_pose=torch.tensor(param['body_pose'], dtype=torch.float32).reshape(1, -1),
#                     betas=torch.tensor(param['shape'], dtype=torch.float32).unsqueeze(0),
#                     transl=torch.tensor(param['trans'], dtype=torch.float32).unsqueeze(0),
#                     expression=torch.tensor(param['expr'], dtype=torch.float32).unsqueeze(0),
#                     leye_pose=torch.tensor(param['leye_pose'], dtype=torch.float32).unsqueeze(0),
#                     reye_pose=torch.tensor(param['reye_pose'], dtype=torch.float32).unsqueeze(0)
#                 )
#                 verts = smplx_output.vertices[0].cpu().numpy()
#                 faces = smplx_model.faces

#                 mesh = o3d.geometry.TriangleMesh()
#                 mesh.vertices = o3d.utility.Vector3dVector(verts)
#                 mesh.triangles = o3d.utility.Vector3iVector(faces)
#                 mesh.compute_vertex_normals()

#                 o3d.visualization.draw_geometries([mesh])



