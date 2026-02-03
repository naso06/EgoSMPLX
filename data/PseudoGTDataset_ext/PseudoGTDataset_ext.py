import os
import os.path as osp
import numpy as np
import torch
from humandata import HumanDataset
import smplx
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pdb import set_trace
from pathlib import Path



class PseudoGTDataset_ext(HumanDataset):
  
    def __init__(self, transform, data_split, data_dir='/media/cv1/SeagateHDD2TB/EgoPW/human_data_train_ext',
                 id_roster=None, return_id_onehot=False, use_cache=False):
        super(PseudoGTDataset_ext, self).__init__(transform, data_split)

        # 설정
        self.data_dir = data_dir
        self.img_shape = (1024, 1280)
        self.annot_path = osp.join(self.data_dir, 'humandata.npz')
        self.annot_path_cache = osp.join(self.data_dir, 'cache', 'humandata_cache.npz')
        self.use_cache = bool(use_cache)
        self.return_id_onehot = bool(return_id_onehot)

        if not osp.exists(self.annot_path):
            raise FileNotFoundError(f"Annotation file not found: {self.annot_path}")

      
        if self.use_cache and osp.exists(self.annot_path_cache):
            print(f"[{self.__class__.__name__}] Loading cache from {self.annot_path_cache}")
            self.datalist = self.load_cache(self.annot_path_cache)
           
            ids = {it.get('id_str') for it in self.datalist if 'id_str' in it}
            self.IDs = sorted([i for i in ids if i is not None])
            self.ID_label2idx = {sid: i for i, sid in enumerate(self.IDs)}
            self.num_ids = len(self.IDs)
            print(f"[{self.__class__.__name__}] Loaded {len(self.datalist)} samples from cache.")
            return

     
        print(f"[{self.__class__.__name__}] Loading annotations from {self.annot_path}")
        data = np.load(self.annot_path, allow_pickle=True)

        image_paths = data['image_path']
        bbox_xywh = data['bbox_xywh']
        keypoints2d = data['keypoints2d_smplx']
        keypoints2d_mask = data['keypoints2d_smplx_mask']
        keypoints3d = data['keypoints3d_smplx']
        keypoints3d_mask = data['keypoints3d_smplx_mask']
        smplx_params = data['smplx']
        self.cam_param = {}

     
        def _extract_subject_id(p: str) -> str:
           
            parts = Path(p).parts
            for marker in ('train', 'test', 'val', 'validation'):
                if marker in parts:
                    i = parts.index(marker)
                    if i + 1 < len(parts):
                        return parts[i + 1]
            # fallback: imgs -> scene -> ID
            try:
                return Path(p).parents[2].name
            except Exception:
               
                return Path(p).parent.name

       
        id_list = []
        abs_paths = []
        for ip in image_paths:
         
            full = ip if osp.isabs(ip) else osp.join(self.data_dir, ip)
            abs_paths.append(full)
            id_list.append(_extract_subject_id(full))

        if id_roster is not None:
            self.IDs = list(id_roster)
        else:
            self.IDs = sorted(set(id_list))
        self.ID_label2idx = {sid: i for i, sid in enumerate(self.IDs)}
        self.num_ids = len(self.IDs)

        def _onehot(idx: int, n: int) -> np.ndarray:
            v = np.zeros((n,), dtype=np.float32)
            v[idx] = 1.0
            return v

    
        self.datalist = []
        for full_img, bbox, kp2d, kp2d_mask, kp3d, kp3d_mask, smplx_d in zip(
            abs_paths, bbox_xywh, keypoints2d, keypoints2d_mask, keypoints3d, keypoints3d_mask, smplx_params
        ):
            sid = _extract_subject_id(full_img)
            sid_idx = self.ID_label2idx[sid]

            item = {
                'img_path': full_img,               
                'img_shape': self.img_shape,
                'bbox': bbox,
                'keypoints2d': kp2d,
                'keypoints2d_mask': kp2d_mask,
                'keypoints3d': kp3d,
                'keypoints3d_mask': kp3d_mask,
                'smplx_param': {
                    'root_pose': smplx_d.get('global_orient', np.zeros((3,))),
                    'body_pose': smplx_d.get('body_pose', np.zeros((69,))).reshape(23, 3)[:21, :],
                    'shape':     smplx_d.get('betas', np.zeros((10,))),
                    'trans':     smplx_d.get('trans', np.zeros((3,))),
                    'leye_pose': smplx_d.get('leye_pose', np.zeros((1, 3))),
                    'reye_pose': smplx_d.get('reye_pose', np.zeros((1, 3))),
              
                    'expr':      smplx_d.get('expr', np.zeros((10,), dtype=np.float32)),
                },
                'smpl_param': {
                    'global_orient': np.zeros((1, 3)),
                    'body_pose':     smplx_d.get('body_pose', np.zeros((69,))).reshape(23, 3),
                    'shape':         smplx_d.get('betas', np.zeros((10,))),
                },
                'joint_img':   kp2d,
                'joint_valid': kp2d_mask.reshape(-1, 1),
                'joint_cam':   None,
                'lhand_bbox':  None,
                'rhand_bbox':  None,
                'face_bbox':   None,

                'id_str': sid,
                'id_idx': int(sid_idx),
            }

            if self.return_id_onehot:
                item['id_onehot'] = _onehot(sid_idx, self.num_ids)

            

            self.datalist.append(item)
            # self.datalist = self.datalist[:100]   

        
        if self.use_cache:
            os.makedirs(osp.dirname(self.annot_path_cache), exist_ok=True)
            self.save_cache(self.annot_path_cache, self.datalist)

        print(f"[{self.__class__.__name__}] IDs({self.num_ids}): {self.IDs}")
        print(f"[{self.__class__.__name__}] Loaded {len(self.datalist)} samples.")

    def visualize_mesh(self, smplx_model_path, sample_idx=0, gender='neutral'):
          
            smplx_model = smplx.SMPLX(
                model_path=smplx_model_path,
                gender=gender,
                batch_size=1,
                use_pca=False,
                create_global_orient=True,
                create_body_pose=True,
                create_betas=True,
                create_transl=True,
                create_expression=True,
                create_left_eye_pose=True,
                create_right_eye_pose=True,
            ).to('cpu')

            param = self.datalist[sample_idx]['smplx_param']
            with torch.no_grad():
                smplx_output = smplx_model(
                    global_orient=torch.tensor(param['root_pose'], dtype=torch.float32).unsqueeze(0),
                    body_pose=torch.tensor(param['body_pose'], dtype=torch.float32).reshape(1, -1),
                    betas=torch.tensor(param['shape'], dtype=torch.float32).unsqueeze(0),
                    transl=torch.tensor(param['trans'], dtype=torch.float32).unsqueeze(0),
                    expression=torch.tensor(param['expr'], dtype=torch.float32).unsqueeze(0),
                    leye_pose=torch.tensor(param['leye_pose'], dtype=torch.float32).unsqueeze(0),
                    reye_pose=torch.tensor(param['reye_pose'], dtype=torch.float32).unsqueeze(0)
                )
                verts = smplx_output.vertices[0].cpu().numpy()
                faces = smplx_model.faces

                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(verts)
                mesh.triangles = o3d.utility.Vector3iVector(faces)
                mesh.compute_vertex_normals()

                o3d.visualization.draw_geometries([mesh])

    def visualize_img_and_joint_separate(self, sample_idx=0):
            sample = self.datalist[sample_idx]
        
         
            img_path = osp.join(self.data_dir, sample['img_path'])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            mo2cap2_to_smplx = {
                "neck": 7, "right_shoulder": 9, "right_elbow": 11, "right_wrist": 13,
                "left_shoulder": 8, "left_elbow": 10, "left_wrist": 12,
                "right_hip": 2, "right_knee": 4, "right_ankle": 6, "right_foot": 19,
                "left_hip": 1, "left_knee": 3, "left_ankle": 5, "left_foot": 16
            }
            joint_names = [
                "neck", "right_shoulder", "right_elbow", "right_wrist",
                "left_shoulder", "left_elbow", "left_wrist",
                "right_hip", "right_knee", "right_ankle", "right_foot",
                "left_hip", "left_knee", "left_ankle", "left_foot"
            ]
            mapping_indices = [mo2cap2_to_smplx[name] for name in joint_names]

            joints_3d_full = sample['joint_cam']  # (137, 3)
            joints_3d = joints_3d_full[mapping_indices]  # (15, 3)

            skeleton_chain = [
                [0, 1, 2, 3],
                [0, 4, 5, 6],
                [1, 7, 8, 9, 10],
                [4, 11, 12, 13, 14],
                [7, 11]
            ]

            fig = plt.figure(figsize=(12, 6))
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.imshow(img)
            ax1.axis('off')
            ax1.set_title("Input Image")

            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            x, y, z = joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2]
            ax2.scatter(x, y, z, c='r', s=20)

            for i, (xi, yi, zi) in enumerate(joints_3d):
                ax2.text(xi, yi, zi, str(i), fontsize=8, color='blue')

            for chain in skeleton_chain:
                chain_pts = joints_3d[chain]
                ax2.plot(chain_pts[:, 0], chain_pts[:, 1], chain_pts[:, 2], c='black', linewidth=1)

            ax2.set_xlabel("X (m)")
            ax2.set_ylabel("Y (m)")
            ax2.set_zlabel("Z (m)")
            ax2.set_title("3D Joint Cam with SMPLX Skeleton")
            ax2.view_init(elev=20, azim=-60)
            plt.tight_layout()
            plt.show()


