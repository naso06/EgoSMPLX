import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.smpler_x import PositionNet, HandRotationNet, FaceRegressor, BoxNet, HandRoI, BodyRotationNet, GlobalHandPositionNet, HandTokenRegressor
# from tokenhmr.lib.models.heads.token_head import SMPLTokenDecoderHead
from nets.loss import CoordLoss, ParamLoss, CELoss
from utils.human_models import smpl_x, smpl
from utils.transforms import rot6d_to_axis_angle, restore_bbox
from config import cfg
import math
import copy
from mmpose.models import build_posenet
from mmcv import Config
from FishEyeCalibrated import FishEyeCameraCalibrated
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pdb import set_trace
# from common.utils.vis import render_mesh_fisheye
from scipy.spatial import cKDTree
from data import humandata
import os
from omegaconf import OmegaConf
from DPoserX.run.tester.wholebody.smplify import DPoser
from torch.autograd import Function


        
# from main.transformer_utils.mmpose.models.backbones.ViT_DINO import vit_large
# from main.transformer_utils.mmpose.models.heads.RAFTDepthNormalDPTDecoder5 import RAFTDepthNormalDPT5
# from mmengine.config import Config

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None
    

bone_pairs = [
    [0, 1], [1, 2], [2, 3],         # right arm
    [0, 4], [4, 5], [5, 6],         # left arm
    [1, 7], [7, 8], [8, 9], [9,10], # right leg
    [4,11], [11,12], [12,13], [13,14], # left leg
]

mo2cap2_to_smplx = {
    "neck": 7,
    "right_shoulder": 9,
    "right_elbow": 11,
    "right_wrist": 13,
    "left_shoulder": 8,
    "left_elbow": 10,
    "left_wrist": 12,
    "right_hip": 2,
    "right_knee": 4,
    "right_ankle": 6,
    "right_foot": 19,
    "left_hip": 1,
    "left_knee": 3,
    "left_ankle": 5,
    "left_foot": 16,
}

mo2cap2_joint_names = [
    "neck","right_shoulder","right_elbow","right_wrist",
    "left_shoulder","left_elbow","left_wrist",
    "right_hip","right_knee","right_ankle","right_foot",
    "left_hip","left_knee","left_ankle","left_foot",
]
mo2cap2_chain = [
    [0, 1, 2, 3],
    [0, 4, 5, 6],
    [1, 7, 8, 9, 10],
    [4, 11, 12, 13, 14],
    [7, 11],
]


class Model(nn.Module):
    def __init__(self, encoder, body_position_net, body_rotation_net, box_net, hand_position_net, hand_roi_net,
                 hand_rotation_net, face_regressor, mode):
        super(Model, self).__init__()

        if getattr(cfg, 'fisheye_camera_path', False):
            fisheye_camera_path = cfg.fisheye_camera_path
        else:
            fisheye_camera_path = '/home/cv1/works/SMPLer-X/main/fisheye.calibration_01_12.json' 
        self.fisheye_camera = FishEyeCameraCalibrated(fisheye_camera_path)
           
        # body
        self.encoder = encoder

        self.box_net = box_net
        
        # hand
        self.hand_roi_net = hand_roi_net
        self.hand_position_net = hand_position_net
        self.hand_regressor = hand_rotation_net
        self.neck = [self.box_net, self.hand_roi_net]
        # face
        self.face_regressor = face_regressor

        self.smplx_layer = copy.deepcopy(smpl_x.layer['neutral']).cuda()
        self.smpl_layer  = copy.deepcopy(smpl.layer['neutral']).cuda()  # SMPL(6890 verts)

        self.coord_loss = CoordLoss()
        self.param_loss = ParamLoss()
        self.ce_loss = CELoss()

        self.body_num_joints = len(smpl_x.pos_joint_part['body'])
        self.hand_joint_num = len(smpl_x.pos_joint_part['rhand'])

        self.body_position_net = body_position_net
        

       
        if getattr(cfg, 'use_smpl', False):
            
            self.body_regressor = body_rotation_net
            self.head = [self.body_position_net, self.body_regressor,
                    self.hand_position_net, self.hand_regressor, 
                    self.face_regressor]
            
            if getattr(cfg, 'use_dposerx', False):
                if mode == 'train':
                    self.dposer_x = DPoser(batch_size=cfg.train_batch_size, config_path=cfg.dposerx_cfg_path).cuda()
                    self.trainable_modules = [self.encoder, self.dposer_x, self.body_position_net, self.body_regressor]
                    #self.trainable_modules = [self.encoder, self.body_position_net, self.body_regressor]
                else:
                    self.dposer_x = DPoser(batch_size=cfg.test_batch_size, config_path=cfg.dposerx_cfg_path).cuda()
            
            else:
                self.trainable_modules = [self.encoder, self.body_position_net, self.body_regressor]

        else:
            
            self.body_regressor = body_rotation_net
            self.head = [self.body_position_net, self.body_regressor,
                    self.hand_position_net, self.hand_regressor, 
                    self.face_regressor]

            if getattr(cfg, 'use_dposerx', False):
                if mode == 'train':
                    self.dposer_x = DPoser(batch_size=cfg.train_batch_size, config_path=cfg.dposerx_cfg_path).cuda()

                    self.trainable_modules = [self.encoder, self.dposer_x, self.body_position_net, self.body_regressor, 
                                    self.box_net, self.hand_position_net,
                                    self.hand_roi_net, self.hand_regressor]
                
                else:
                    self.dposer_x = DPoser(batch_size=cfg.test_batch_size, config_path=cfg.dposerx_cfg_path).cuda()
            
            else:
                self.trainable_modules = [self.encoder, self.body_position_net, self.body_regressor, 
                                    self.box_net, self.hand_position_net,
                                    self.hand_roi_net, self.hand_regressor, self.face_regressor]
                
                           
        self.special_trainable_modules = []
        
        # --- ID adversarial head (optional) ---
        self.id_adv_lambda = float(getattr(cfg, 'id_adv_lambda', 0.0))   # e.g., 0.5
        self.num_subjects  = int(getattr(cfg, 'num_subjects', 0))        # ID 클래스 수
        if self.id_adv_lambda > 0 and self.num_subjects > 0:
            # img_feat의 채널 차원 = encoder 출력 채널(보통 cfg.feat_dim과 일치)
            self.id_head = nn.Linear(cfg.feat_dim, self.num_subjects)
            self.id_loss_fn = nn.BCEWithLogitsLoss()
            # 학습 모듈에 포함(선택)
            self.trainable_modules.append(self.id_head)
            self.head.append(self.id_head) 

        # backbone:
        param_bb = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        # neck 
        param_neck = 0
        for module in self.neck:
            param_neck += sum(p.numel() for p in module.parameters() if p.requires_grad)
        # head
        param_head = 0
        for module in self.head:
            param_head += sum(p.numel() for p in module.parameters() if p.requires_grad)

        param_net = param_bb + param_neck + param_head

                # ---- 파라미터 개수 출력 ----
        total_params = sum(p.numel() for p in self.parameters())
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        def _m(x):
            return x / 1e6  # million 단위

        print("========== SMPLer-X Parameter Counts ==========")
        print(f"Backbone (encoder, trainable) : {param_bb:,}  ({_m(param_bb):.3f} M)")
        print(f"Neck (box + hand_roi)        : {param_neck:,}  ({_m(param_neck):.3f} M)")
        print(f"Head (body/hand/face 등)     : {param_head:,}  ({_m(param_head):.3f} M)")
        print("----------------------------------------------")
        print(f"Total trainable              : {param_net:,}  ({_m(param_net):.3f} M)")
        print(f"Total (all params)           : {total_params:,}  ({_m(total_params):.3f} M)")
        print("==============================================")


        

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

        mapped_pixels = []
        for r in rays_flat:
            try:
                uv = self.fisheye_camera.world2camera(np.array([r]))
                mapped_pixels.append(uv[0])
            except:
                mapped_pixels.append([-1, -1])

        self.mapped_pixels = np.array(mapped_pixels, dtype=np.float32).reshape(self.undist_h, self.undist_w, 2)

    def _get_subject_scene_dir(self, img_path):
        """
        img_path: .../{subject}/{scene}/imgs/filename 형태라고 가정하고
        마지막 'imgs' 기준으로 바로 앞의 두 디렉토리(subject, scene)를 리턴.
        실패하면 빈 문자열 반환.
        """
        if not isinstance(img_path, str):
            return ""
        path = img_path.replace("\\", "/")
        parts = path.split("/")

        # 마지막 'imgs' 위치 찾기
        idx = None
        for j in range(len(parts) - 1, -1, -1):
            if parts[j] == "imgs":
                idx = j
                break

        if idx is None or idx < 2:
            return ""

        subject = parts[idx - 2]
        scene = parts[idx - 1]
        return os.path.join(subject, scene)

        
    def _save_activation_heatmap(
        self,
        img_3chw: torch.Tensor,
        feat_3chw: torch.Tensor,
        save_path: str,
        alpha: float = 0.45,
        # 새 인자들: 원래 토큰 그리드와 잘려나간 칸 수
        orig_grid_hw=(16, 16),           # (H0, W0) = 원래 16x16
        crop=(0, 2, 0, 2),               # (top, right, bottom, left) = 좌우 2칸
        upsample: str = "nearest"        # 토큰 경계 또렷: 'nearest' 권장
    ):
        """
        img_3chw : [3,H,W]
        feat_3chw: [C,h,w]  (현재 16x12 토큰 그리드로 재배치된 feature)
        """
        import os
        import matplotlib.pyplot as plt
        import torch.nn.functional as F

        # 1) 채널 평균 → (h,w)
        hm = feat_3chw.detach().float().mean(0)  # [h,w]  (※ .values 쓰지 마세요)

        # 2) (선택) 간단 정규화
        hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-6)

        # 3) 잘려나간 칸만큼 패딩해서 원래 16x16 위치로 복원 + per-pixel alpha 마스크
        H0, W0 = orig_grid_hw         # ex) 16, 16
        t, r, b, l = crop             # ex) (0,2,0,2)
        cam_pad   = torch.zeros((H0, W0), dtype=hm.dtype, device=hm.device)
        alpha_pad = torch.zeros_like(cam_pad)  # 패딩 영역은 투명(0)
        cam_pad[t:H0-b, l:W0-r]   = hm
        alpha_pad[t:H0-b, l:W0-r] = 1.0

        # 4) 이미지 해상도로 업샘플 (히트맵/알파)
        H_img, W_img = img_3chw.shape[-2], img_3chw.shape[-1]
        cam_up = F.interpolate(cam_pad[None, None], size=(H_img, W_img),
                            mode=upsample, align_corners=False if upsample!="nearest" else None)[0, 0].cpu().numpy()
        a_up   = F.interpolate(alpha_pad[None, None], size=(H_img, W_img),
                            mode="nearest")[0, 0].cpu().numpy()  # 알파는 nearest 권장

        # 5) 원본 오버레이 (잘려나간 영역은 자동으로 투명)
        img = img_3chw.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy()
        plt.figure(figsize=(6, 4.5))
        plt.imshow(img)
        plt.imshow(cam_up, cmap="viridis", alpha=alpha * a_up)  # per-pixel alpha
        plt.axis("off"); plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

    def visualize_hand_bboxes_on_input(self, img_batch, lhand_bbox, rhand_bbox,
                                       save_dir, meta_info=None):
        """
        img_batch : [B, 3, H, W] (inputs['img_ori'])
        lhand_bbox, rhand_bbox : [B, 4] (xyxy, input_img_shape 기준)
        """
        os.makedirs(save_dir, exist_ok=True)

        # [B, H, W, C] 로 변환
        img_np = img_batch.detach().cpu().numpy().transpose(0, 2, 3, 1)

        B = img_np.shape[0]
        for i in range(B):
            img = img_np[i]

            # 이미지 값 범위에 따라 0~255 uint8 로 변환
            if img.max() <= 1.0:
                img = (img * 255.0).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            vis = img.copy()

            h, w = vis.shape[:2]

            def _draw_box(box, color, label_text=None):
                box = box.detach().cpu().numpy()
                x1, y1, x2, y2 = box

                x1 = int(np.clip(x1, 0, w - 1))
                x2 = int(np.clip(x2, 0, w - 1))
                y1 = int(np.clip(y1, 0, h - 1))
                y2 = int(np.clip(y2, 0, h - 1))

                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                if label_text is not None:
                    cv2.putText(vis, label_text, (x1, max(0, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                                lineType=cv2.LINE_AA)

            # 왼손: 초록, 오른손: 파랑
            _draw_box(lhand_bbox[i], (0, 255, 0), 'L-hand')
            _draw_box(rhand_bbox[i], (255, 0, 0), 'R-hand')

            # # 파일 이름 결정 (img_path 있으면 거기서 베이스 이름 사용)
            # if meta_info is not None and 'img_path' in meta_info:
            #     base = meta_info['img_path'][i]
            #     if not isinstance(base, str):
            #         base = f"{i:06d}.png"
            #     name = os.path.splitext(os.path.basename(base))[0]
            # else:
            #     name = f"{i:06d}"

            # out_path = os.path.join(save_dir, f"{name}_hand_bbox.png")
            # # OpenCV는 BGR 을 기대하므로 RGB -> BGR 로 뒤집어서 저장
            # cv2.imwrite(out_path, vis[:, :, ::-1])

            img_path_i = None
            sub_rel = ""
            if meta_info is not None and 'img_path' in meta_info:
                img_path_i = meta_info['img_path'][i]
                if isinstance(img_path_i, str):
                    sub_rel = self._get_subject_scene_dir(img_path_i)
                    name = os.path.splitext(os.path.basename(img_path_i))[0]
                else:
                    name = f"{i:06d}"
            else:
                name = f"{i:06d}"

            if sub_rel:
                cur_dir = os.path.join(save_dir, sub_rel)
            else:
                cur_dir = save_dir

            os.makedirs(cur_dir, exist_ok=True)

            out_path = os.path.join(cur_dir, f"{name}_hand_bbox.png")
            cv2.imwrite(out_path, vis[:, :, ::-1])


    def get_camera_trans(self, cam_param):
        # camera translation
        t_xy = cam_param[:, :2]
        gamma = torch.sigmoid(cam_param[:, 2])  # apply sigmoid to make it positive
        k_value = torch.FloatTensor([math.sqrt(cfg.focal[0] * cfg.focal[1] * cfg.camera_3d_size * cfg.camera_3d_size / (
                cfg.input_img_shape[0] * cfg.input_img_shape[1]))]).cuda().view(-1)
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[:, None]), 1)
        return cam_trans


    def undistort_tensor_image(self, tensor_img, fisheye_camera: FishEyeCameraCalibrated):
        """
        tensor_img: [B, C, H, W] - torch.Tensor (range 0~1 or 0~255)
        return: undistorted image in torch.Tensor [B, C, H, W]
        """
        device = tensor_img.device
        B, C, H, W = tensor_img.shape
        tensor_img_np = tensor_img.detach().cpu().numpy().transpose(0, 2, 3, 1)  # [B, H, W, C]

        undistorted_np = []
        for i in range(B):
            img = (tensor_img_np[i] * 255).astype(np.uint8) if tensor_img_np[i].max() <= 1 else tensor_img_np[i].astype(np.uint8)

            # grid 생성
            x = np.linspace(-1, 1, W)
            y = np.linspace(-1, 1, H)
            xx, yy = np.meshgrid(x, y)
            z = 1.0  # FOV에 해당하는 focal length 기준
            rays = np.stack([xx, yy, np.full_like(xx, z)], axis=-1)
            rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
            rays_flat = rays.reshape(-1, 3)

            mapped_pixels = []
            for r in rays_flat:
                try:
                    uv = self.fisheye_camera.world2camera(np.array([r]))  # (1, 2)
                    mapped_pixels.append(uv[0])
                except:
                    mapped_pixels.append([-1, -1])
            mapped_pixels = np.array(mapped_pixels, dtype=np.float32).reshape(H, W, 2)

            undistorted_img = cv2.remap(
                img,
                mapped_pixels[:, :, 0],
                mapped_pixels[:, :, 1],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            undistorted_np.append(undistorted_img)

        undistorted_np = np.stack(undistorted_np).astype(np.float32) / 255.0
        undistorted_tensor = torch.from_numpy(undistorted_np).permute(0, 3, 1, 2).to(device)
        return undistorted_tensor
    



    def visualize_body_img_comparison(self, original, undistorted):
        """
        original, undistorted: [B, C, H, W] torch.Tensor
        """
        import matplotlib.pyplot as plt

        B = original.shape[0]
        for i in range(B):
            orig_np = original[i].detach().cpu().permute(1, 2, 0).numpy()
            undist_np = undistorted[i].detach().cpu().permute(1, 2, 0).numpy()

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(np.clip(orig_np, 0, 1))
            plt.title('Original (Fisheye)')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(np.clip(undist_np, 0, 1))
            plt.title('Undistorted')
            plt.axis('off')

            plt.tight_layout()
            plt.show()

    def get_coord(self, root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape, expr, cam_trans, mode):
        batch_size = root_pose.shape[0]
        # set_trace()
        zero_pose = torch.zeros((1, 3)).float().cuda().repeat(batch_size, 1)  # eye poses
        
        # if getattr(cfg, 'use_token_decoder', False):
        #     pad = torch.zeros(body_pose.size(0), 6, dtype=body_pose.dtype, device=body_pose.device) # (B, 2, 3)
        #     body_pose = torch.cat((body_pose, pad), 1)
        #     output = self.smpl_layer(betas=shape, body_pose=body_pose, global_orient=root_pose)
        
        # else:
        output = self.smplx_layer(betas=shape, body_pose=body_pose, global_orient=root_pose, right_hand_pose=rhand_pose,
                            left_hand_pose=lhand_pose, jaw_pose=jaw_pose, leye_pose=zero_pose,
                            reye_pose=zero_pose, expression=expr)

        mesh_cam = output.vertices + cam_trans[:, None, :]  # ← 최종 mesh_cam은 SMPL 기준

        # smplx decoder 사용
        
        # camera-centered 3D coordinate

        mesh_cam = output.vertices
        if mode == 'test' and cfg.testset == 'AGORA':  # use 144 joints for AGORA evaluation
            joint_cam = output.joints
        else:
            joint_cam = output.joints[:, smpl_x.joint_idx, :]
            # set_trace()
            


        B, J, _ = joint_cam.shape
        # set_trace()
        joint_cam_translation =joint_cam.detach() + cam_trans[:, None, :]

        joint_cam_flat = (joint_cam_translation).view(-1, 3)  # (B*J, 3)

        # (B*J, 3) → (B*J, 2): fisheye projection (image 좌표계)
        joint_proj_flat = self.fisheye_camera.world2camera_pytorch(joint_cam_flat)  # (B*J, 2)

        # (B*J, 2) → (B, J, 2)
        joint_proj = joint_proj_flat.view(B, J, 2)
        
        # root-relative 3D coordinates
        root_cam = joint_cam[:, smpl_x.root_joint_idx, None, :]
        joint_cam = joint_cam - root_cam
        
        mesh_cam = mesh_cam + cam_trans[:, None, :]  # for rendering
        joint_cam_wo_ra = joint_cam.clone()
        
        # 2. heatmap 좌표계로 변환
        joint_proj[..., 0] = joint_proj[..., 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2] # x: w 기준
        joint_proj[..., 1] = joint_proj[..., 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1] # y: h 기준
        
   
        # left hand root (left wrist)-relative 3D coordinatese
        lhand_idx = smpl_x.joint_part['lhand']
        lhand_cam = joint_cam[:, lhand_idx, :]
        lwrist_cam = joint_cam[:, smpl_x.lwrist_idx, None, :]
        lhand_cam = lhand_cam - lwrist_cam
        joint_cam = torch.cat((joint_cam[:, :lhand_idx[0], :], lhand_cam, joint_cam[:, lhand_idx[-1] + 1:, :]), 1)

        # right hand root (right wrist)-relative 3D coordinatese
        rhand_idx = smpl_x.joint_part['rhand']
        rhand_cam = joint_cam[:, rhand_idx, :]
        rwrist_cam = joint_cam[:, smpl_x.rwrist_idx, None, :]
        rhand_cam = rhand_cam - rwrist_cam
        joint_cam = torch.cat((joint_cam[:, :rhand_idx[0], :], rhand_cam, joint_cam[:, rhand_idx[-1] + 1:, :]), 1)

        # face root (neck)-relative 3D coordinates
        face_idx = smpl_x.joint_part['face']
        face_cam = joint_cam[:, face_idx, :]
        neck_cam = joint_cam[:, smpl_x.neck_idx, None, :]
        face_cam = face_cam - neck_cam
        joint_cam = torch.cat((joint_cam[:, :face_idx[0], :], face_cam, joint_cam[:, face_idx[-1] + 1:, :]), 1)


        return joint_proj, joint_cam, joint_cam_wo_ra, mesh_cam

    def generate_mesh_gt(self, targets, mode):
        if 'smplx_mesh_cam' in targets:
            return targets['smplx_mesh_cam']
        nums = [3, 63, 45, 45, 3]
        accu = []
        temp = 0
        for num in nums:
            temp += num
            accu.append(temp)
        pose = targets['smplx_pose']
        root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose = \
            pose[:, :accu[0]], pose[:, accu[0]:accu[1]], pose[:, accu[1]:accu[2]], pose[:, accu[2]:accu[3]], pose[:,
                                                                                                             accu[3]:
                                                                                                             accu[4]]
        # print(lhand_pose)
        shape = targets['smplx_shape']
        expr = targets['smplx_expr']
        cam_trans = targets['smplx_cam_trans']

        # final output
        joint_proj, joint_cam, joint_cam_wo_ra, mesh_cam = self.get_coord(root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape,
                                                         expr, cam_trans, mode)

        return mesh_cam

    def bbox_split(self, bbox):
        # bbox:[bs, 3, 3]
        lhand_bbox_center, rhand_bbox_center, face_bbox_center = \
            bbox[:, 0, :2], bbox[:, 1, :2], bbox[:, 2, :2]
        return lhand_bbox_center, rhand_bbox_center, face_bbox_center
    
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

    def remap_joints_forward(self, joint_orig, tree, dst_coords):
        """
        joint_orig: (N, 2) — 원본 이미지 기준 joint
        tree: KDTree of mapped_pixels (src → dst)
        dst_coords: 보정된 이미지 좌표

        반환: 보정된 joint 위치 (N, 2)
        """
        dist, idx = tree.query(joint_orig, k=1)
        joint_remapped = dst_coords[idx]
        return joint_remapped.astype(np.float32)
    
   
    def draw_joint_lines(self, img_np, joints_2d, joint_names, chains, color=(0,255,0)):
        """
        img_np: (H,W,3) numpy array (uint8)
        joints_2d: (N,2) float (xy)
        joint_names: 이름 리스트
        chains: 각 limb의 joint 인덱스 chain(순서대로 연결)
        color: 선 색
        """
        # joint 찍기
        for x, y in joints_2d:
            x_int, y_int = int(round(x)), int(round(y))
            if 0 <= x_int < img_np.shape[1] and 0 <= y_int < img_np.shape[0]:
                cv2.circle(img_np, (x_int, y_int), 3, color, -1)

        # 선 그리기
        for chain in chains:
            for i in range(len(chain)-1):
                idx1, idx2 = chain[i], chain[i+1]
                x1, y1 = joints_2d[idx1]
                x2, y2 = joints_2d[idx2]
                pt1 = (int(round(x1)), int(round(y1)))
                pt2 = (int(round(x2)), int(round(y2)))
                if all(0 <= pt < img_np.shape[1] for pt in pt1) and all(0 <= pt < img_np.shape[0] for pt in pt1) and \
                all(0 <= pt < img_np.shape[1] for pt in pt2) and all(0 <= pt < img_np.shape[0] for pt in pt2):
                    cv2.line(img_np, pt1, pt2, (255, 128, 0), 2)  # 선 색은 예시 (주황)

        return img_np
    
    def visualize_mesh_on_image(self, img_np, mesh_cam, color=(0, 255, 0)):
        """
        img_np: (H, W, 3) numpy array (uint8)
        mesh_cam: (N, 3) torch.Tensor - 카메라 좌표계의 mesh vertices
        """
        # numpy로 변환
        mesh_cam_np = mesh_cam.detach().cpu().numpy()

        # 3D mesh를 fisheye projection을 통해 2D로 투영
        mesh_proj_2d = self.fisheye_camera.world2camera_pytorch(
            torch.from_numpy(mesh_cam_np).float().cuda()
        ).detach().cpu().numpy()  # (N, 2)

        # 각 vertex를 이미지 위에 표시
        for x, y in mesh_proj_2d:
            x_int, y_int = int(round(x)), int(round(y))
            if 0 <= x_int < img_np.shape[1] and 0 <= y_int < img_np.shape[0]:
                cv2.circle(img_np, (x_int, y_int), 1, color, -1)

        return img_np

    def forward(self, inputs, targets, meta_info, mode):
        # body_img = F.interpolate(inputs['img_ori'], cfg.input_body_shape)
        # body_img = inputs['img']
        #     # 0. Fisheye 보정 수행
        

        # # 0.5 시각화 (선택)
        # self.visualize_body_img_comparison(body_img, body_img_undistorted)
        body_img = inputs['img_ori']
        # set_trace()
        # 1. Encoder
        ## img_feat: [bs, 1280, 16, 12], task_tokens: [bs, 31, 1280]
        img_feat, task_tokens = self.encoder(body_img)  # task_token:[bs, N, c]
        shape_token, cam_token, expr_token, jaw_pose_token, hand_token, body_pose_token = \
            task_tokens[:, 0], task_tokens[:, 1], task_tokens[:, 2], task_tokens[:, 3], task_tokens[:, 4:6], task_tokens[:, 6:]


        # 1-1. ID Regressor

        # person_id = self.id_regressor(id_token)
        
        # set_trace()
        # 2. Body Regressor
        if not getattr(cfg, 'use_token_decoder', False):
            body_joint_hm, body_joint_img = self.body_position_net(img_feat)
            root_pose, body_pose, shape, cam_param, = self.body_regressor(body_pose_token, shape_token, cam_token, body_joint_img.detach())
            

       


        # set_trace()
        root_pose = rot6d_to_axis_angle(root_pose)
        body_pose = rot6d_to_axis_angle(body_pose.reshape(-1, 6)).reshape(body_pose.shape[0], -1)  # (N, J_R*3)
        # set_trace()
    
        # 3. Hand and Face BBox Estimation
        lhand_bbox_center, lhand_bbox_size, rhand_bbox_center, rhand_bbox_size, face_bbox_center, face_bbox_size = self.box_net(img_feat, body_joint_hm.detach())
        lhand_bbox = restore_bbox(lhand_bbox_center, lhand_bbox_size, cfg.input_hand_shape[1] / cfg.input_hand_shape[0], 2.0).detach()  # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space
        rhand_bbox = restore_bbox(rhand_bbox_center, rhand_bbox_size, cfg.input_hand_shape[1] / cfg.input_hand_shape[0], 2.0).detach()  # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space
        face_bbox = restore_bbox(face_bbox_center, face_bbox_size, cfg.input_face_shape[1] / cfg.input_face_shape[0], 1.5).detach()  # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space

        # 4. Differentiable Feature-level Hand Crop-Upsample
        # hand_feat: list, [bsx2, c, cfg.output_hm_shape[1]*scale, cfg.output_hm_shape[2]*scale]
        hand_feat = self.hand_roi_net(img_feat, lhand_bbox, rhand_bbox)  # hand_feat: flipped left hand + right hand

        # 5. Hand/Face Regressor
        # hand regressor
        _, hand_joint_img = self.hand_position_net(hand_feat)  # (2N, J_P, 3)
        hand_pose = self.hand_regressor(hand_feat, hand_joint_img.detach())
        hand_pose = rot6d_to_axis_angle(hand_pose.reshape(-1, 6)).reshape(hand_feat.shape[0], -1)  # (2N, J_R*3)

        # restore flipped left hand joint coordinates
        batch_size = hand_joint_img.shape[0] // 2
        lhand_joint_img = hand_joint_img[:batch_size, :, :]
        lhand_joint_img = torch.cat((cfg.output_hand_hm_shape[2] - 1 - lhand_joint_img[:, :, 0:1], lhand_joint_img[:, :, 1:]), 2)
        rhand_joint_img = hand_joint_img[batch_size:, :, :]
        # restore flipped left hand joint rotations
        batch_size = hand_pose.shape[0] // 2
        lhand_pose = hand_pose[:batch_size, :].reshape(-1, len(smpl_x.orig_joint_part['lhand']), 3)
        lhand_pose = torch.cat((lhand_pose[:, :, 0:1], -lhand_pose[:, :, 1:3]), 2).view(batch_size, -1)
        rhand_pose = hand_pose[batch_size:, :]

        

        # # 5. Hand regressor without ROI
        # hand_joint_hm, hand_joint_coord_global = self.hand_position_net(img_feat)  # (B, 2J_all, 3)

        # B = img_feat.size(0)
        # J_all = self.hand_joint_num                      # 20 (keypoints/joints)
        # J_p = len(smpl_x.orig_joint_part['lhand'])       # 보통 15 (pose parameter joints)

        # # split left/right (each is (B, J_all, 3))
        # lhand_joint_img = hand_joint_coord_global[:, :J_all, :]
        # rhand_joint_img = hand_joint_coord_global[:, J_all:, :]

        # # --- 중요: pose에 쓰는 joint만 선택 ---
        # # 대부분 ordering이 [wrist(0), finger 15개(1~15), tips 4개(16~19)] 라서
        # # wrist/tips 제외하고 finger 15개만 쓰면 됨.
        # lhand_joint_img_pose = lhand_joint_img[:, 1:1+J_p, :]   # (B, J_p, 3)
        # rhand_joint_img_pose = rhand_joint_img[:, 1:1+J_p, :]   # (B, J_p, 3)

        # hand_joint_coord = torch.stack([lhand_joint_img_pose, rhand_joint_img_pose], dim=1)  # (B, 2, J_p, 3)

        # token+joint -> 6D  (B, 2, J_p, 6)
        # hand_pose_6d = self.hand_regressor(hand_token, hand_joint_coord.detach())

        # 6D -> axis-angle  (B, 2, J_p, 3)
        # hand_pose_aa = rot6d_to_axis_angle(hand_pose_6d.reshape(-1, 6)).view(B, 2, J_p, 3)

        # (B, J_p*3) == (B, 45) 로 맞춰짐
        # lhand_pose = hand_pose_aa[:, 0].reshape(B, -1)
        # rhand_pose = hand_pose_aa[:, 1].reshape(B, -1)


        # face regressor
        # set_trace()
        expr, jaw_pose = self.face_regressor(expr_token, jaw_pose_token)
        jaw_pose = rot6d_to_axis_angle(jaw_pose)

        # final output
        joint_proj, joint_cam, joint_cam_wo_ra, mesh_cam = self.get_coord(root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape, expr, cam_param, mode)
       
    
        pose = torch.cat((root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose), 1)
        # set_trace()


        # set_trace()
        B      = body_pose.size(0)
        device = body_pose.device
        dtype  = body_pose.dtype
        expr_zeros = torch.zeros(B, 100, device=device, dtype=dtype) 

        pose_experssion = torch.cat((body_pose, lhand_pose, rhand_pose, jaw_pose, expr_zeros), 1)
        joint_img = torch.cat((body_joint_img, lhand_joint_img, rhand_joint_img), 1)

        # if mode == 'test' and 'smplx_pose' in targets:
        #     mesh_pseudo_gt = self.generate_mesh_gt(targets, mode)

        

        if mode == 'train':
            loss = {}

            if mode == 'train' and getattr(self, 'id_head', None) is not None:
                id_loss_weight = getattr(cfg, 'id_loss_weight', 1.0)
                # img_feat: [B, C, H, W]  -> GAP로 [B, C]
                g = F.adaptive_avg_pool2d(img_feat, 1).flatten(1)


                # GRL 통과(부호 반전 + 스케일링)
                # g_rev = GradReverse.apply(g, self.id_adv_lambda)

                # set_trace()
                id_logits = self.id_head(g)   # [B, num_subjects]

                # 타깃: 정수 인덱스 형태 권장 (DataLoader에서 meta_info['id_idx']로 전달)
                id_idx = meta_info['id_idx'].long()                 # [B]
                id_target = F.one_hot(id_idx, num_classes=self.num_subjects).float()  # [B, K]
                loss['id_adv'] = self.id_loss_fn(id_logits, id_target) * id_loss_weight

            # loss functions
            

            smplx_kps_3d_weight = getattr(cfg, 'smplx_kps_3d_weight', 1.0)
            smplx_kps_3d_weight = getattr(cfg, 'smplx_kps_weight', smplx_kps_3d_weight) # old config

            smplx_kps_2d_weight = getattr(cfg, 'smplx_kps_2d_weight', 1.0)
            net_kps_2d_weight = getattr(cfg, 'net_kps_2d_weight', 1.0)

            smplx_pose_weight = getattr(cfg, 'smplx_pose_weight', 1.0)
            smplx_shape_weight = getattr(cfg, 'smplx_loss_weight', 1.0)

            dposer_x_weight = getattr(cfg, 'dposer_x_weight', 1.0)
            # smplx_orient_weight = getattr(cfg, 'smplx_orient_weight', smplx_pose_weight) # if not specified, use the same weight as pose
    

            # do not supervise root pose if original agora json is used


            ###  SMPL parameter regression ###
            if getattr(cfg, 'agora_fix_global_orient_transl', False):
                # loss['smplx_pose'] = self.param_loss(pose, targets['smplx_pose'], meta_info['smplx_pose_valid'])[:, 3:] * smplx_pose_weight
                if hasattr(cfg, 'smplx_orient_weight'):
                    smplx_orient_weight = getattr(cfg, 'smplx_orient_weight')
                    loss['smplx_orient'] = self.param_loss(pose, targets['smplx_pose'], meta_info['smplx_pose_valid'])[:, :3] * smplx_orient_weight

                loss['smplx_pose'] = self.param_loss(pose, targets['smplx_pose'], meta_info['smplx_pose_valid']) * smplx_pose_weight
                # loss['smplx_lhand_pose'] = self.param_loss(lhand_pose, targets['smplx_lhand_pose'], meta_info['smplx_pose_valid'][:, 70:116]) * smplx_pose_weight
                # loss['smplx_lhand_pose'] = self.param_loss(lhand_pose, targets['smplx_lhand_pose'], meta_info['smplx_pose_valid'][:, 116:162]) * smplx_pose_weight

            else:
                loss['smplx_pose'] = self.param_loss(pose, targets['smplx_pose'], meta_info['smplx_pose_valid']) * smplx_pose_weight
                # loss['smplx_lhand_pose'] = self.param_loss(lhand_pose, targets['smplx_lhand_pose'], meta_info['smplx_pose_valid'][:, 70:116]) * smplx_pose_weight
                # loss['smplx_lhand_pose'] = self.param_loss(lhand_pose, targets['smplx_lhand_pose'], meta_info['smplx_pose_valid'][:, 116:162]) * smplx_pose_weight


            loss['smplx_shape'] = self.param_loss(shape, targets['smplx_shape'],
                                                  meta_info['smplx_shape_valid'][:, None]) * smplx_shape_weight 
            loss['smplx_expr'] = self.param_loss(expr, targets['smplx_expr'], meta_info['smplx_expr_valid'][:, None])

            loss['dposerx'] = self.dposer_x(pose_experssion) * dposer_x_weight
            # supervision for keypoints3d wo/ ra
            # loss['joint_cam'] = self.coord_loss(joint_cam_wo_ra, targets['joint_cam'], meta_info['joint_valid'] * meta_info['is_3D'][:, None, None]) * smplx_kps_3d_weight
            # supervision for keypoints3d w/ ra


            loss['smplx_joint_cam'] = self.coord_loss(joint_cam, targets['smplx_joint_cam'], meta_info['smplx_joint_valid']) * smplx_kps_3d_weight


            # loss['bone_length'] = self.compute_bone_length_loss(
            #     joint_cam[:, [mo2cap2_to_smplx[name] for name in mo2cap2_joint_names], :],
            #     targets['smplx_joint_cam'][:, [mo2cap2_to_smplx[name] for name in mo2cap2_joint_names], :],
            #     bone_pairs,
            #     weight=bone_loss_weight
            # )
            if not (meta_info['lhand_bbox_valid'] == 0).all():
                loss['lhand_bbox'] = (self.coord_loss(lhand_bbox_center, targets['lhand_bbox_center'], meta_info['lhand_bbox_valid'][:, None]) +
                                    self.coord_loss(lhand_bbox_size, targets['lhand_bbox_size'], meta_info['lhand_bbox_valid'][:, None]))
            if not (meta_info['rhand_bbox_valid'] == 0).all():
                loss['rhand_bbox'] = (self.coord_loss(rhand_bbox_center, targets['rhand_bbox_center'], meta_info['rhand_bbox_valid'][:, None]) +
                                    self.coord_loss(rhand_bbox_size, targets['rhand_bbox_size'], meta_info['rhand_bbox_valid'][:, None]))
            if not (meta_info['face_bbox_valid'] == 0).all():
                loss['face_bbox'] = (self.coord_loss(face_bbox_center, targets['face_bbox_center'], meta_info['face_bbox_valid'][:, None]) +
                                 self.coord_loss(face_bbox_size, targets['face_bbox_size'], meta_info['face_bbox_valid'][:, None]))
            
            # if (meta_info['face_bbox_valid'] == 0).all():
            #     out = {}
            targets['original_joint_img'] = targets['joint_img'].clone()
            targets['original_smplx_joint_img'] = targets['smplx_joint_img'].clone()
            # out['original_joint_proj'] = joint_proj.clone()
            if not (meta_info['lhand_bbox_valid'] + meta_info['rhand_bbox_valid'] == 0).all():

                # change hand target joint_img and joint_trunc according to hand bbox (cfg.output_hm_shape -> downsampled hand bbox space)
                for part_name, bbox in (('lhand', lhand_bbox), ('rhand', rhand_bbox)):
                    for coord_name, trunc_name in (('joint_img', 'joint_trunc'), ('smplx_joint_img', 'smplx_joint_trunc')):
                        x = targets[coord_name][:, smpl_x.joint_part[part_name], 0]
                        y = targets[coord_name][:, smpl_x.joint_part[part_name], 1]
                        z = targets[coord_name][:, smpl_x.joint_part[part_name], 2]
                        trunc = meta_info[trunc_name][:, smpl_x.joint_part[part_name], 0]

                        x -= (bbox[:, None, 0] / cfg.input_body_shape[1] * cfg.output_hm_shape[2])
                        x *= (cfg.output_hand_hm_shape[2] / (
                                (bbox[:, None, 2] - bbox[:, None, 0]) / cfg.input_body_shape[1] * cfg.output_hm_shape[
                            2]))
                        y -= (bbox[:, None, 1] / cfg.input_body_shape[0] * cfg.output_hm_shape[1])
                        y *= (cfg.output_hand_hm_shape[1] / (
                                (bbox[:, None, 3] - bbox[:, None, 1]) / cfg.input_body_shape[0] * cfg.output_hm_shape[
                            1]))
                        z *= cfg.output_hand_hm_shape[0] / cfg.output_hm_shape[0]
                        trunc *= ((x >= 0) * (x < cfg.output_hand_hm_shape[2]) * (y >= 0) * (
                                y < cfg.output_hand_hm_shape[1]))

                        coord = torch.stack((x, y, z), 2)
                        trunc = trunc[:, :, None]
                        targets[coord_name] = torch.cat((targets[coord_name][:, :smpl_x.joint_part[part_name][0], :], coord,
                                                        targets[coord_name][:, smpl_x.joint_part[part_name][-1] + 1:, :]),
                                                        1)
                        meta_info[trunc_name] = torch.cat((meta_info[trunc_name][:, :smpl_x.joint_part[part_name][0], :],
                                                        trunc,
                                                        meta_info[trunc_name][:, smpl_x.joint_part[part_name][-1] + 1:,
                                                        :]), 1)

                # change hand projected joint coordinates according to hand bbox (cfg.output_hm_shape -> hand bbox space)
                for part_name, bbox in (('lhand', lhand_bbox), ('rhand', rhand_bbox)):
                    x = joint_proj[:, smpl_x.joint_part[part_name], 0]
                    y = joint_proj[:, smpl_x.joint_part[part_name], 1]

                    x -= (bbox[:, None, 0] / cfg.input_body_shape[1] * cfg.output_hm_shape[2])
                    x *= (cfg.output_hand_hm_shape[2] / (
                            (bbox[:, None, 2] - bbox[:, None, 0]) / cfg.input_body_shape[1] * cfg.output_hm_shape[2]))
                    y -= (bbox[:, None, 1] / cfg.input_body_shape[0] * cfg.output_hm_shape[1])
                    y *= (cfg.output_hand_hm_shape[1] / (
                            (bbox[:, None, 3] - bbox[:, None, 1]) / cfg.input_body_shape[0] * cfg.output_hm_shape[1]))

                    coord = torch.stack((x, y), 2)
                    trans = []
                    for bid in range(coord.shape[0]):
                        mask = meta_info['joint_trunc'][bid, smpl_x.joint_part[part_name], 0] == 1
                        if torch.sum(mask) == 0:
                            trans.append(torch.zeros((2)).float().cuda())
                        else:
                            trans.append((-coord[bid, mask, :2] + targets['joint_img'][:, smpl_x.joint_part[part_name], :][
                                                                bid, mask, :2]).mean(0))
                    trans = torch.stack(trans)[:, None, :]
                    coord = coord + trans  # global translation alignment
                    joint_proj = torch.cat((joint_proj[:, :smpl_x.joint_part[part_name][0], :], coord,
                                            joint_proj[:, smpl_x.joint_part[part_name][-1] + 1:, :]), 1)

            if not (meta_info['face_bbox_valid'] == 0).all():
                # change face projected joint coordinates according to face bbox (cfg.output_hm_shape -> face bbox space)
                coord = joint_proj[:, smpl_x.joint_part['face'], :]
                trans = []
                for bid in range(coord.shape[0]):
                    mask = meta_info['joint_trunc'][bid, smpl_x.joint_part['face'], 0] == 1
                    if torch.sum(mask) == 0:
                        trans.append(torch.zeros((2)).float().cuda())
                    else:
                        trans.append((-coord[bid, mask, :2] + targets['joint_img'][:, smpl_x.joint_part['face'], :][bid,
                                                            mask, :2]).mean(0))
                trans = torch.stack(trans)[:, None, :]
                coord = coord + trans  # global translation alignment
                joint_proj = torch.cat((joint_proj[:, :smpl_x.joint_part['face'][0], :], coord,
                                        joint_proj[:, smpl_x.joint_part['face'][-1] + 1:, :]), 1)
           # print("joint_proj:", joint_proj.shape)
           # print("coord_gt  :", targets['joint_img'][:, :, :2].shape)
           # print("valid     :", meta_info['joint_trunc'].shape)
            # set_trace()


            loss['joint_proj'] = self.coord_loss(joint_proj, targets['smplx_joint_img'][:, :, :2], meta_info['smplx_joint_trunc']) * smplx_kps_2d_weight
            


            # loss['joint_img'] = self.coord_loss(joint_img, smpl_x.reduce_joint_set(targets['joint_img']),
            #                                     smpl_x.reduce_joint_set(meta_info['joint_trunc']), meta_info['is_3D']) * net_kps_2d_weight
            # set_trace()

            # PositionNet
            loss['smplx_joint_img'] = self.coord_loss(joint_img, smpl_x.reduce_joint_set(targets['smplx_joint_img']),
                                                     smpl_x.reduce_joint_set(meta_info['smplx_joint_trunc'])) * net_kps_2d_weight

            return loss
        else:

            save_dir = os.path.join(cfg.result_dir, 'vis_test_result')
            os.makedirs(save_dir, exist_ok=True)
            B = body_img.shape[0]

            if getattr(cfg, 'vis_feature', False):
                base_feat_dir = os.path.join(cfg.result_dir, 'vis_test_result', 'feature')
                B = body_img.shape[0]

                for i in range(B):
                    img_path_i = None
                    if 'img_path' in meta_info:
                        img_path_i = meta_info['img_path'][i]

                    # dataset/.../{subject}/{scene}/imgs/... 에서 subject/scene 뽑기
                    sub_rel = self._get_subject_scene_dir(img_path_i) if isinstance(img_path_i, str) else ""

                    if sub_rel:
                        feat_dir = os.path.join(base_feat_dir, sub_rel)
                    else:
                        feat_dir = base_feat_dir

                    os.makedirs(feat_dir, exist_ok=True)

                    if isinstance(img_path_i, str):
                        base = os.path.basename(img_path_i)
                        name = os.path.splitext(base)[0]
                    else:
                        name = f"{i:06d}"

                    save_path = os.path.join(feat_dir, f"{name}_feat.png")
                    # body_img: [B,3,H,W], img_feat: [B,C,h,w]
                    self._save_activation_heatmap(body_img[i], img_feat[i], save_path)


            # change hand output joint_img according to hand bbox
            for part_name, bbox in (('lhand', lhand_bbox), ('rhand', rhand_bbox)):
                joint_img[:, smpl_x.pos_joint_part[part_name], 0] *= (
                        ((bbox[:, None, 2] - bbox[:, None, 0]) / cfg.input_body_shape[1] * cfg.output_hm_shape[2]) /
                        cfg.output_hand_hm_shape[2])
                joint_img[:, smpl_x.pos_joint_part[part_name], 0] += (
                        bbox[:, None, 0] / cfg.input_body_shape[1] * cfg.output_hm_shape[2])
                joint_img[:, smpl_x.pos_joint_part[part_name], 1] *= (
                        ((bbox[:, None, 3] - bbox[:, None, 1]) / cfg.input_body_shape[0] * cfg.output_hm_shape[1]) /
                        cfg.output_hand_hm_shape[1])
                joint_img[:, smpl_x.pos_joint_part[part_name], 1] += (
                        bbox[:, None, 1] / cfg.input_body_shape[0] * cfg.output_hm_shape[1])

            # change input_body_shape to input_img_shape
            for bbox in (lhand_bbox, rhand_bbox, face_bbox):
                bbox[:, 0] *= cfg.input_img_shape[1] / cfg.input_body_shape[1]
                bbox[:, 1] *= cfg.input_img_shape[0] / cfg.input_body_shape[0]
                bbox[:, 2] *= cfg.input_img_shape[1] / cfg.input_body_shape[1]
                bbox[:, 3] *= cfg.input_img_shape[0] / cfg.input_body_shape[0]


            if getattr(cfg, 'vis_hand_bbox', False):
                hand_bbox_dir = os.path.join(save_dir, 'hand_bbox')
                os.makedirs(hand_bbox_dir, exist_ok=True)
                self.visualize_hand_bboxes_on_input(
                    body_img, lhand_bbox, rhand_bbox,
                    hand_bbox_dir, meta_info
                )

           #  for i in range(B):
                # set_trace()
                # img_np = inputs['img_vis'][i].detach().cpu().numpy()  # [H, W, C]
                # img_np = np.clip(img_np, 0, 1) * 255.0
                # img_np = img_np.astype(np.uint8).copy()

                # selected_joint_indices = list(mo2cap2_to_smplx.values()) 
                # joints_2d = joint_proj[i][selected_joint_indices].detach().cpu().numpy()  # [N_valid, 2]
                # for x, y in joints_2d:
                #     x_int, y_int = int(round(x)), int(round(y))
                #     if 0 <= x_int < img_np.shape[1] and 0 <= y_int < img_np.shape[0]:
                #         cv2.circle(img_np, (x_int, y_int), 3, (0, 255, 0), -1)

                # joint 시각화
                # img_np = self.draw_joint_lines(img_np, joints_2d, mo2cap2_joint_names, mo2cap2_chain)

                # mesh 시각화
                # img_np = self.visualize_mesh_on_image(img_np, mesh_cam[i])


                # save_path = os.path.join(save_dir, f'proj_{i:03d}.png')
                # cv2.imwrite(save_path, img_np[:, :, ::-1])

            # test output
            out = {}
            out['img'] = inputs['img_ori']
            # out['img_vis'] = inputs['img_vis']
            out['joint_img'] = joint_img

            out['smplx_joint_proj'] = joint_proj
            out['smplx_mesh_cam'] = mesh_cam
            out['smplx_root_pose'] = root_pose
            out['smplx_body_pose'] = body_pose
            out['smplx_lhand_pose'] = lhand_pose
            out['smplx_rhand_pose'] = rhand_pose
            out['smplx_jaw_pose'] = jaw_pose
            out['smplx_shape'] = shape
            out['smplx_expr'] = expr
            # out['cam_trans'] = cam_trans
            # out['lhand_bbox'] = lhand_bbox
            # out['rhand_bbox'] = rhand_bbox
            # out['face_bbox'] = face_bbox
            if 'smplx_shape' in targets:
                out['smplx_shape_target'] = targets['smplx_shape']
            if 'img_path' in meta_info:
                out['img_path'] = meta_info['img_path']
            if 'smplx_joint_cam' in targets:
                out['gt_joint'] = targets['smplx_joint_cam']
            # if 'smplx_pose' in targets:
            #     out['smplx_mesh_cam_pseudo_gt'] = mesh_pseudo_gt
            if 'smplx_joint_img' in targets:
                out['smplx_joint_img'] = targets['smplx_joint_img']
            if 'smplx_mesh_cam' in targets:
                out['smplx_mesh_cam_target'] = targets['smplx_mesh_cam']
            if 'smpl_mesh_cam' in targets:
                out['smpl_mesh_cam_target'] = targets['smpl_mesh_cam']
            if 'bb2img_trans' in meta_info:
                out['bb2img_trans'] = meta_info['bb2img_trans']
            if 'gt_smplx_transl' in meta_info:
                out['gt_smplx_transl'] = meta_info['gt_smplx_transl']

            return out

def init_weights(m):
    try:
        if type(m) == nn.ConvTranspose2d:
            nn.init.normal_(m.weight, std=0.001)
        elif type(m) == nn.Conv2d:
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.BatchNorm2d:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)
    except AttributeError:
        pass

def reinit_cam_out(m):
    """cam_out만 랜덤 초기화"""
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)
        nn.init.constant_(m.bias, 0)


def get_model(mode):

    # body
    vit_cfg = Config.fromfile(cfg.encoder_config_file)
    vit = build_posenet(vit_cfg.model)

   
    body_position_net = PositionNet('body', feat_dim=cfg.feat_dim)
    body_rotation_net = BodyRotationNet(feat_dim=cfg.feat_dim)
    box_net = BoxNet(feat_dim=cfg.feat_dim)

    # hand

    ## no-ROI
    # hand_position_net = GlobalHandPositionNet(feat_dim = cfg.feat_dim)
    # hand_rotation_net = HandTokenRegressor()

    hand_position_net = PositionNet('hand', feat_dim=cfg.feat_dim)
    hand_roi_net = HandRoI(feat_dim=cfg.feat_dim, upscale=cfg.upscale)
    hand_rotation_net = HandRotationNet('hand', feat_dim=cfg.feat_dim)

    # face
    face_regressor = FaceRegressor(feat_dim=cfg.feat_dim)

    if mode == 'train':
        # body
        if not getattr(cfg, 'random_init', False):
            encoder_pretrained_model = torch.load(cfg.encoder_pretrained_model_path)['state_dict']
            vit.load_state_dict(encoder_pretrained_model, strict=False)
            print(f"Initialize encoder from {cfg.encoder_pretrained_model_path}")
        else:
            print('Random init!!!!!!!')

        if getattr(cfg, 'use_smpl', False):
            body_position_net.apply(init_weights)
            body_rotation_net.apply(init_weights)

        else:

            body_position_net.apply(init_weights)
            body_rotation_net.apply(init_weights)
            
            box_net.apply(init_weights)

     

            # hand
            hand_position_net.apply(init_weights)
            hand_roi_net.apply(init_weights)
            hand_rotation_net.apply(init_weights)

            # face
            face_regressor.apply(init_weights)

        

    encoder = vit.backbone

    model = Model(encoder, body_position_net, body_rotation_net, box_net, hand_position_net, hand_roi_net, hand_rotation_net,
                  face_regressor, mode)
  
    return model
