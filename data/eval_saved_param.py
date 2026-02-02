#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from pdb import set_trace
from typing import Optional


# -------------------------
# Make sure SMPLer-X root is on PYTHONPATH
# data/eval_saved_param.py 기준: ROOT = .../SMPLer-X
# -------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# SMPLer-X 내부 smplx 래퍼
# 경로: SMPLer-X/common/utils/smplx/smplx/...
from common.utils.smplx import smplx  # <-- SMPLer-X repo structure


from pathlib import Path

def resolve_pred_path(img_path: str,
                      pred_root: str,
                      pred_format: str,
                      src_root: Optional[str] = None) -> str:
    """
    img_path: gt npz에 들어있는 이미지 경로(대개 src_root 기준 상대경로 or 절대경로)
    pred_root: 예측 pkl들이 저장된 루트
    pred_format: 예: "{img_path}_smplx.pkl"
    src_root: img_path가 절대경로일 때, 이 prefix를 떼고 pred_root에 붙이기 위해 사용
    """
    # 1) img_path -> 상대 경로로 정규화
    p = Path(img_path)
    if src_root is not None:
        try:
            p = p.relative_to(Path(src_root))
        except Exception:
            # src_root와 무관하면 그대로 사용
            pass

    # 2) format 치환
    #    사용자가 "{img_path}_smplx.pkl"처럼 주므로 suffix까지 포함될 수 있음
    rel_no_ext = str(p)  # 확장자 포함/미포함은 pred_format이 결정
    pred_rel = pred_format.format(img_path=rel_no_ext)

    # 3) pred_root 붙이기
    return str(Path(pred_root) / pred_rel)


# -------------------------
# Geometry utils
# -------------------------
def batch_rigid_align_by_root(pred, gt, root_idx=0):
    """
    Root-align (translation only) by subtracting root joint.
    pred, gt: (N, J, 3)
    """
    pred = pred - pred[:, [root_idx], :]
    gt = gt - gt[:, [root_idx], :]
    return pred, gt


def compute_similarity_transform(X, Y):
    """
    Procrustes: find sR,t to align X to Y.
    X, Y: (J,3)
    Returns aligned X_hat: (J,3)
    """
    X = X.T
    Y = Y.T

    muX = X.mean(axis=1, keepdims=True)
    muY = Y.mean(axis=1, keepdims=True)
    X0 = X - muX
    Y0 = Y - muY

    varX = np.sum(X0 ** 2)
    K = X0 @ Y0.T

    U, _, Vt = np.linalg.svd(K)
    V = Vt.T
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U @ V.T))
    R = V @ Z @ U.T

    s = np.trace(R @ K) / (varX + 1e-9)
    t = muY - s * (R @ muX)

    X_hat = (s * R @ X) + t
    return X_hat.T


def pa_align_batch(pred, gt):
    """
    pred, gt: (N, J, 3)
    return pred_aligned: (N,J,3)
    """
    out = []
    for p, g in zip(pred, gt):
        out.append(compute_similarity_transform(p, g))
    return np.stack(out, axis=0).astype(np.float32)


def mpjpe(pred, gt):
    # (N,J,3)
    return np.linalg.norm(pred - gt, axis=-1).mean() * 1000.0


def mpvpe(pred_v, gt_v):
    # (N,V,3)
    return np.linalg.norm(pred_v - gt_v, axis=-1).mean() * 1000.0


def pa_mpjpe(pred, gt):
    pred_aligned = pa_align_batch(pred, gt)
    return mpjpe(pred_aligned, gt), pred_aligned


def pa_mpvpe(pred_v, gt_v):
    pred_aligned = pa_align_batch(pred_v, gt_v)
    return mpvpe(pred_aligned, gt_v), pred_aligned


# -------------------------
# Minimal SMPL-X loader (no config.py / human_models.py)
# -------------------------
class MinimalSMPLX:
    def __init__(self, human_model_path: str, device: str = "cuda", gender: str = "neutral",
                 flat_hand_mean: bool = False):
        """
        human_model_path should contain 'smplx' folder and the pkl/npy files:
          - smplx/SMPLX_to_J14.pkl
          - smplx/MANO_SMPLX_vertex_ids.pkl
          - smplx/SMPL-X__FLAME_vertex_ids.npy
        """
        self.human_model_path = human_model_path
        self.device = torch.device(device)
        self.gender = gender.lower()

        layer_arg = dict(
            create_global_orient=False,
            create_body_pose=False,
            create_left_hand_pose=False,
            create_right_hand_pose=False,
            create_jaw_pose=False,
            create_leye_pose=False,
            create_reye_pose=False,
            create_betas=False,
            create_expression=False,
            create_transl=False,
        )
        create_kwargs = dict(use_pca=False, use_face_contour=True, **layer_arg)
        if flat_hand_mean:
            create_kwargs["flat_hand_mean"] = True

        # smplx.create expects model root path, then model type folder inside it
        # SMPLer-X uses 'smplx' folder under human_model_path
        self.layer = smplx.create(
            self.human_model_path, "smplx",
            gender="NEUTRAL" if self.gender == "neutral" else self.gender.upper(),
            **create_kwargs
        ).to(self.device)
        self.layer.eval()

        self.vertex_num = 10475
        self.face = self.layer.faces

        # load hand vertex indices (MANO mapping)
        with open(os.path.join(self.human_model_path, "smplx", "MANO_SMPLX_vertex_ids.pkl"), "rb") as f:
            self.hand_vertex_idx = pickle.load(f, encoding="latin1")
        # expected keys: 'left_hand', 'right_hand' OR 'left', 'right' depending on file
        # we normalize later.

        # face vertex indices (FLAME mapping used by SMPLer-X)
        self.face_vertex_idx = np.load(os.path.join(self.human_model_path, "smplx", "SMPL-X__FLAME_vertex_ids.npy"))

        # J regressor (used for pelvis / wrist / neck alignment exactly like utils.human_models.smpl_x)
        self.J_regressor = self.layer.J_regressor.detach().cpu().numpy().astype(np.float32)
        self.J_regressor_idx = {"pelvis": 0, "lwrist": 20, "rwrist": 21, "neck": 12}
        # original hand joint regressors (match human_models.py / humandata.py)
        self.orig_hand_regressor = self._make_hand_regressor()

        # joint indices for root-align: SMPL-X joints include pelvis at 0 usually
        self.root_joint_idx = 0


    @staticmethod
    def _get_hand_vertex_indices(hand_vertex_idx):
        """
        Normalize possible key names in MANO_SMPLX_vertex_ids.pkl
        """
        if isinstance(hand_vertex_idx, dict):
            if "left" in hand_vertex_idx and "right" in hand_vertex_idx:
                return np.array(hand_vertex_idx["left"], dtype=np.int64), np.array(hand_vertex_idx["right"], dtype=np.int64)
            if "left_hand" in hand_vertex_idx and "right_hand" in hand_vertex_idx:
                return np.array(hand_vertex_idx["left_hand"], dtype=np.int64), np.array(hand_vertex_idx["right_hand"], dtype=np.int64)

        raise KeyError("MANO_SMPLX_vertex_ids.pkl dict keys not recognized. Expected left/right or left_hand/right_hand.")


    def _make_hand_regressor(self):
        """Build 21-joint hand regressors used in humandata.py (see human_models.py)."""
        reg = self.J_regressor.astype(np.float32)  # (J,V)
        V = reg.shape[1]
        def onehot(vid):
            v = np.zeros((1, V), dtype=np.float32)
            v[0, int(vid)] = 1.0
            return v
        lhand = np.concatenate((
            reg[[20, 37, 38, 39], :],
            onehot(5361),
            reg[[25, 26, 27], :],
            onehot(4933),
            reg[[28, 29, 30], :],
            onehot(5058),
            reg[[34, 35, 36], :],
            onehot(5169),
            reg[[31, 32, 33], :],
            onehot(5286),
        ), axis=0)
        rhand = np.concatenate((
            reg[[21, 52, 53, 54], :],
            onehot(8079),
            reg[[40, 41, 42], :],
            onehot(7669),
            reg[[43, 44, 45], :],
            onehot(7794),
            reg[[49, 50, 51], :],
            onehot(7905),
            reg[[46, 47, 48], :],
            onehot(8022),
        ), axis=0)
        return {'left': lhand.astype(np.float32), 'right': rhand.astype(np.float32)}

    @torch.no_grad()
    def forward(self, params: dict):
        """
        params keys (all torch tensors on device):
          betas: (N,10)
          global_orient: (N,3) axis-angle
          body_pose: (N,63) axis-angle
          jaw_pose: (N,3)
          leye_pose: (N,3)
          reye_pose: (N,3)
          left_hand_pose: (N,45)
          right_hand_pose: (N,45)
          expression: (N,10)
          transl: (N,3)
        returns:
          joints: (N, J, 3)
          vertices: (N, V, 3)
          hand_vertices: dict with left/right
        """
        out = self.layer(
            betas=params["betas"],
            global_orient=params["global_orient"],
            body_pose=params["body_pose"],
            jaw_pose=params["jaw_pose"],
            leye_pose=params["leye_pose"],
            reye_pose=params["reye_pose"],
            left_hand_pose=params["left_hand_pose"],
            right_hand_pose=params["right_hand_pose"],
            expression=params["expression"],
            transl=params["transl"],
        )
        joints = out.joints  # (N,J,3)
        verts = out.vertices # (N,V,3)

        l_vid, r_vid = self._get_hand_vertex_indices(self.hand_vertex_idx)
        l_verts = verts[:, l_vid, :]
        r_verts = verts[:, r_vid, :]

        return joints, verts, {"left": l_verts, "right": r_verts}


# -------------------------
# IO helpers
# -------------------------
def load_humandata_npz(npz_path: str):
    """
    humandata.npz 내부 구조는 프로젝트마다 다를 수 있어서,
    자주 쓰는 key들을 유연하게 처리.
    """
    data = np.load(npz_path, allow_pickle=True)

    # image path list key candidate
    img_keys = ["img_path", "img_paths", "image_path", "image_paths", "imgname", "img_name", "img"]
    img_paths = None
    for k in img_keys:
        if k in data:
            img_paths = data[k]
            break
    if img_paths is None:
        raise KeyError(f"Cannot find image path list in npz. Tried keys: {img_keys}. Available: {list(data.keys())}")

    # SMPL-X GT params key candidates
    # many pipelines store a dict array or separate arrays.
    # We'll support two common styles:
    # 1) data['smplx'] is an array of dicts (N,)
    # 2) separate arrays: global_orient, body_pose, ... etc
    gt = {}

    if "smplx" in data:
        # could be dict list
        smplx_arr = data["smplx"]
        if smplx_arr.dtype == object:
            # list of dicts
            gt["smplx_dicts"] = smplx_arr
        else:
            # unknown
            raise TypeError("npz['smplx'] exists but not object dtype; please adapt loader.")
    else:
        # separate arrays
        for k in ["shape", "global_orient", "body_pose", "jaw_pose", "leye_pose", "reye_pose",
                  "left_hand_pose", "right_hand_pose", "expression", "transl"]:
            if k in data:
                gt[k] = data[k]

    return img_paths, gt, data


def gt_params_for_index(gt, idx, device, force_shape10_zero=False):
    """
    Return dict of torch tensors for SMPL-X layer.
    GT는 humandata.npz에서 가져옴.
    """
    # helper: pick first existing key
    def pick(d, keys):
        for k in keys:
            if k in d:
                return k
        return None

    out = {}

    if "smplx_dicts" in gt:
        d = gt["smplx_dicts"][idx].item() if hasattr(gt["smplx_dicts"][idx], "item") else gt["smplx_dicts"][idx]

        # betas / shape
        betas_key = pick(d, ["betas", "shape"])
        if betas_key is None:
            raise KeyError("GT dict missing betas/shape")
        betas = np.asarray(d[betas_key], dtype=np.float32).reshape(-1)

        # transl / trans
        transl_key = pick(d, ["transl", "trans"])
        if transl_key is None:
            raise KeyError("GT dict missing transl/trans")
        transl = np.asarray(d[transl_key], dtype=np.float32).reshape(-1)

        # remaining required keys (use defaults if your GT doesn't have them)
        def getv(key, default):
            return np.asarray(d[key], dtype=np.float32).reshape(-1) if key in d else np.asarray(default, dtype=np.float32).reshape(-1)

        out["global_orient"]   = torch.from_numpy(getv("global_orient", np.zeros(3, np.float32))).to(device).view(1, -1)
        out["body_pose"]       = torch.from_numpy(getv("body_pose", np.zeros(63, np.float32))).to(device).view(1, -1)
        out["jaw_pose"]        = torch.from_numpy(getv("jaw_pose", np.zeros(3, np.float32))).to(device).view(1, -1)
        out["leye_pose"]       = torch.from_numpy(getv("leye_pose", np.zeros(3, np.float32))).to(device).view(1, -1)
        out["reye_pose"]       = torch.from_numpy(getv("reye_pose", np.zeros(3, np.float32))).to(device).view(1, -1)
        out["left_hand_pose"]  = torch.from_numpy(getv("left_hand_pose", np.zeros(45, np.float32))).to(device).view(1, -1)
        out["right_hand_pose"] = torch.from_numpy(getv("right_hand_pose", np.zeros(45, np.float32))).to(device).view(1, -1)
        out["expression"]      = torch.from_numpy(getv("expression", np.zeros(10, np.float32))).to(device).view(1, -1)

        if force_shape10_zero:
            out["betas"] = torch.zeros((1,10), dtype=torch.float32, device=device)
        else:
            out["betas"] = torch.from_numpy(betas).to(device).view(1, -1)
            if out["betas"].shape[1] != 10:
                out["betas"] = out["betas"][:, :10]

        out["transl"] = torch.from_numpy(transl).to(device).view(1, -1)

    else:
        # separate arrays case
        # betas/shape
        betas_key = pick(gt, ["betas", "shape"])
        if betas_key is None:
            raise KeyError("GT npz missing betas/shape array")
        betas = np.asarray(gt[betas_key][idx], dtype=np.float32).reshape(-1)

        # transl/trans
        transl_key = pick(gt, ["transl", "trans"])
        if transl_key is None:
            raise KeyError("GT npz missing transl/trans array")
        transl = np.asarray(gt[transl_key][idx], dtype=np.float32).reshape(-1)

        def getarr(key, default):
            return np.asarray(gt[key][idx], dtype=np.float32).reshape(-1) if key in gt else np.asarray(default, dtype=np.float32).reshape(-1)

        out["global_orient"]   = torch.from_numpy(getarr("global_orient", np.zeros(3, np.float32))).to(device).view(1, -1)
        out["body_pose"]       = torch.from_numpy(getarr("body_pose", np.zeros(63, np.float32))).to(device).view(1, -1)
        out["jaw_pose"]        = torch.from_numpy(getarr("jaw_pose", np.zeros(3, np.float32))).to(device).view(1, -1)
        out["leye_pose"]       = torch.from_numpy(getarr("leye_pose", np.zeros(3, np.float32))).to(device).view(1, -1)
        out["reye_pose"]       = torch.from_numpy(getarr("reye_pose", np.zeros(3, np.float32))).to(device).view(1, -1)
        out["left_hand_pose"]  = torch.from_numpy(getarr("left_hand_pose", np.zeros(45, np.float32))).to(device).view(1, -1)
        out["right_hand_pose"] = torch.from_numpy(getarr("right_hand_pose", np.zeros(45, np.float32))).to(device).view(1, -1)
        out["expression"]      = torch.from_numpy(getarr("expression", np.zeros(10, np.float32))).to(device).view(1, -1)

        if force_shape10_zero:
            out["betas"] = torch.zeros((1,10), dtype=torch.float32, device=device)
        else:
            out["betas"] = torch.from_numpy(betas).to(device).view(1, -1)
            if out["betas"].shape[1] != 10:
                out["betas"] = out["betas"][:, :10]

        out["transl"] = torch.from_numpy(transl).to(device).view(1, -1)

    return out



def load_pred_pkl(pred_path: str):
    with open(pred_path, "rb") as f:
        d = pickle.load(f)
    return d


def pred_dict_to_params(d, device, pred_betas_zero=False, expr_dim=10):
    """
    pred pkl dict -> torch params dict (batch=1)
    Expected keys in pkl:
      global_orient, body_pose, jaw_pose, leye_pose, reye_pose,
      left_hand_pose, right_hand_pose, betas, transl, expression(optional)
    """
    out = {}

    def getv(key, default=None):
        if key in d:
            return np.asarray(d[key], dtype=np.float32).reshape(-1)
        if default is None:
            raise KeyError(f"Pred pkl missing key '{key}'")
        return np.asarray(default, dtype=np.float32).reshape(-1)

    out["global_orient"] = torch.from_numpy(getv("global_orient")).to(device).view(1, -1)
    out["body_pose"] = torch.from_numpy(getv("body_pose")).to(device).view(1, -1)
    out["jaw_pose"] = torch.from_numpy(getv("jaw_pose", np.zeros(3, np.float32))).to(device).view(1, -1)
    out["leye_pose"] = torch.from_numpy(getv("leye_pose", np.zeros(3, np.float32))).to(device).view(1, -1)
    out["reye_pose"] = torch.from_numpy(getv("reye_pose", np.zeros(3, np.float32))).to(device).view(1, -1)
    out["left_hand_pose"] = torch.from_numpy(getv("left_hand_pose", np.zeros(45, np.float32))).to(device).view(1, -1)
    out["right_hand_pose"] = torch.from_numpy(getv("right_hand_pose", np.zeros(45, np.float32))).to(device).view(1, -1)

    # betas: either from file or forced zero
    if pred_betas_zero:
        out["betas"] = torch.zeros((1,10), dtype=torch.float32, device=device)
    else:
        out["betas"] = torch.from_numpy(getv("betas")).to(device).view(1, -1)
        if out["betas"].shape[1] != 10:
            out["betas"] = out["betas"][:, :10]

    # expression: default zeros
    out["expression"] = torch.from_numpy(getv("expression", np.zeros(expr_dim, np.float32))).to(device).view(1, -1)
    if out["expression"].shape[1] != expr_dim:
        out["expression"] = torch.zeros((1, expr_dim), dtype=torch.float32, device=device)

    out["transl"] = torch.from_numpy(getv("transl", np.zeros(3, np.float32))).to(device).view(1, -1)

    return out


# -------------------------
# Main evaluation
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--humandata", "--gt-npz", dest="humandata", required=True, type=str,
                    help="Path to humandata.npz (GT inside)")
    ap.add_argument("--smplx-model-path", required=True, type=str,
                    help="Path containing smplx model files (the folder that contains 'smplx' subfolder)")
    ap.add_argument("--src-root", required=True, type=str,
                    help="Root of dataset paths used in humandata (to resolve relative img paths if needed)")
    ap.add_argument("--pred-root", required=True, type=str,
                    help="Root folder where predicted pkls are stored")
    ap.add_argument("--pred-format", required=True, type=str,
                    help='Format string for pred pkl path. Use {img_path} placeholder. '
                         'Example: "{img_path}_smplx.pkl" or "{img_path}.pkl"')
    ap.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"])
    ap.add_argument("--skip-missing", action="store_true",
                    help="Skip samples whose pred file is missing")
    ap.add_argument("--pred-betas-zero", action="store_true",
                    help="Force pred betas to zeros (10D)")
    ap.add_argument("--gt-betas-zero", action="store_true",
                    help="Force GT betas to zeros (10D) (usually NOT needed)")
    ap.add_argument("--expr-dim", type=int, default=10)
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] cuda requested but not available. Falling back to cpu.")
        device = "cpu"

    # load GT
    img_paths, gt, _ = load_humandata_npz(args.humandata)

    # build smplx
    smplx_model = MinimalSMPLX(
        human_model_path=args.smplx_model_path,
        device=device,
        gender="neutral",
        flat_hand_mean=False
    )

    # metrics accum
        # -------------------------
    # 
    # -------------------------
    # Metric computation (match humandata.py exactly)
    # - root-align uses J_regressor pelvis / wrist / neck
    # - MPJPE uses joints regressed from vertices (J_regressor @ vertices), not SMPL-X output joints
    # - hands/face combined metrics are the average of (left, right) means per sample
    # -------------------------
    sum_mpjpe_ra = 0.0
    sum_mpvpe_ra = 0.0
    sum_pa_mpjpe = 0.0
    sum_pa_mpvpe = 0.0

    sum_lh_mpvpe_ra = 0.0
    sum_rh_mpvpe_ra = 0.0
    sum_hand_mpvpe_ra = 0.0

    sum_lh_pa_mpvpe = 0.0
    sum_rh_pa_mpvpe = 0.0
    sum_hand_pa_mpvpe = 0.0

    sum_face_mpvpe_ra = 0.0
    sum_face_pa_mpvpe = 0.0
    # hand joint MPJPE (match humandata.py)
    sum_lh_mpjpe = 0.0
    sum_rh_mpjpe = 0.0
    sum_hand_mpjpe = 0.0
    sum_lh_pa_mpjpe = 0.0
    sum_rh_pa_mpjpe = 0.0
    sum_hand_pa_mpjpe = 0.0


    used = 0
    missed = 0

    # Pre-compute vertex index arrays
    l_vid, r_vid = smplx_model._get_hand_vertex_indices(smplx_model.hand_vertex_idx)
    face_vid = np.array(smplx_model.face_vertex_idx, dtype=np.int64)

    J_reg = smplx_model.J_regressor
    J_idx = smplx_model.J_regressor_idx

    for i, img_path in enumerate(img_paths):
        pred_pkl_path = resolve_pred_path(
            img_path=img_path,
            src_root=args.src_root,
            pred_root=args.pred_root,
            pred_format=args.pred_format,
        )

        if not os.path.exists(pred_pkl_path):
            missed += 1
            if args.skip_missing:
                continue
            raise FileNotFoundError(f"Missing pred pkl: {pred_pkl_path}")

        pred = load_pred_pkl(pred_pkl_path)
        gt_params = gt_params_for_index(
            gt, i,
            device=torch.device(device),
            force_shape10_zero=args.gt_betas_zero
        )

        pred_params = pred_dict_to_params(
            pred,
            device=torch.device(device),
            pred_betas_zero=args.pred_betas_zero,
            expr_dim=args.expr_dim
        )

        # Forward SMPL-X
        pred_joints_t, pred_verts_t, _ = smplx_model.forward(pred_params)
        gt_joints_t, gt_verts_t, _ = smplx_model.forward(gt_params)

        pred_v = pred_verts_t[0].detach().cpu().numpy().astype(np.float32)
        gt_v   = gt_verts_t[0].detach().cpu().numpy().astype(np.float32)

        # ===== Root-align full mesh using pelvis from J_regressor =====
        pred_J_all = J_reg @ pred_v  # (J,3)
        gt_J_all   = J_reg @ gt_v

        pelvis_pred = pred_J_all[J_idx["pelvis"]]
        pelvis_gt   = gt_J_all[J_idx["pelvis"]]

        pred_v_ra = pred_v - pelvis_pred[None, :] + pelvis_gt[None, :]
        d_v = np.linalg.norm(pred_v_ra - gt_v, axis=-1)
        sum_mpvpe_ra += float(d_v.mean())

        # MPJPE uses regressed joints from pelvis-aligned mesh (match humandata.py)
        pred_j_ra = J_reg @ pred_v_ra
        gt_j      = gt_J_all
        d_j = np.linalg.norm(pred_j_ra - gt_j, axis=-1)
        sum_mpjpe_ra += float(d_j.mean())

        # ===== PA alignment =====
        pred_j_pa = compute_similarity_transform(pred_j_ra, gt_j)
        sum_pa_mpjpe += float(np.linalg.norm(pred_j_pa - gt_j, axis=-1).mean())

        pred_v_pa = compute_similarity_transform(pred_v, gt_v)
        sum_pa_mpvpe += float(np.linalg.norm(pred_v_pa - gt_v, axis=-1).mean())

        # ===== Hands (vertex) =====
        pred_lh = pred_v[l_vid, :]
        pred_rh = pred_v[r_vid, :]
        gt_lh   = gt_v[l_vid, :]
        gt_rh   = gt_v[r_vid, :]

        lwrist_pred = pred_J_all[J_idx["lwrist"]]
        rwrist_pred = pred_J_all[J_idx["rwrist"]]
        lwrist_gt   = gt_J_all[J_idx["lwrist"]]
        rwrist_gt   = gt_J_all[J_idx["rwrist"]]

        pred_lh_ra = pred_lh - lwrist_pred[None, :] + lwrist_gt[None, :]
        pred_rh_ra = pred_rh - rwrist_pred[None, :] + rwrist_gt[None, :]

        lh_mean = float(np.linalg.norm(pred_lh_ra - gt_lh, axis=-1).mean())
        rh_mean = float(np.linalg.norm(pred_rh_ra - gt_rh, axis=-1).mean())
        sum_lh_mpvpe_ra += lh_mean
        sum_rh_mpvpe_ra += rh_mean
        sum_hand_mpvpe_ra += (lh_mean + rh_mean) / 2.0

        # PA hands: align each hand separately (match humandata.py)
        pred_lh_pa = compute_similarity_transform(pred_lh, gt_lh)
        pred_rh_pa = compute_similarity_transform(pred_rh, gt_rh)
        lh_pa_mean = float(np.linalg.norm(pred_lh_pa - gt_lh, axis=-1).mean())
        rh_pa_mean = float(np.linalg.norm(pred_rh_pa - gt_rh, axis=-1).mean())
        sum_lh_pa_mpvpe += lh_pa_mean
        sum_rh_pa_mpvpe += rh_pa_mean
        sum_hand_pa_mpvpe += (lh_pa_mean + rh_pa_mean) / 2.0

        # ===== Face (vertex) =====
        pred_face = pred_v[face_vid, :]
        gt_face   = gt_v[face_vid, :]

        neck_pred = pred_J_all[J_idx["neck"]]
        neck_gt   = gt_J_all[J_idx["neck"]]
        pred_face_ra = pred_face - neck_pred[None, :] + neck_gt[None, :]
        sum_face_mpvpe_ra += float(np.linalg.norm(pred_face_ra - gt_face, axis=-1).mean())

        pred_face_pa = compute_similarity_transform(pred_face, gt_face)
        sum_face_pa_mpvpe += float(np.linalg.norm(pred_face_pa - gt_face, axis=-1).mean())


        # ===== Hand joints (MPJPE / PA-MPJPE) =====
        # Regress 21 hand joints from vertices using the same hand regressors as humandata.py
        R_l = smplx_model.orig_hand_regressor['left']   # (21,V)
        R_r = smplx_model.orig_hand_regressor['right']  # (21,V)

        # joints from vertices (21,3)
        gt_lj = (R_l @ gt_v)      # (21,3)
        pred_lj = (R_l @ pred_v)  # (21,3)
        gt_rj = (R_r @ gt_v)
        pred_rj = (R_r @ pred_v)

        # wrist-align predicted hand joints to GT wrist using J_regressor wrists (match humandata.py)
        lw_pred = pred_J_all[J_idx['lwrist']]
        lw_gt   = gt_J_all[J_idx['lwrist']]
        rw_pred = pred_J_all[J_idx['rwrist']]
        rw_gt   = gt_J_all[J_idx['rwrist']]

        pred_lj_ra = pred_lj - lw_pred[None, :] + lw_gt[None, :]
        pred_rj_ra = pred_rj - rw_pred[None, :] + rw_gt[None, :]

        lh_mpjpe = float(np.linalg.norm(pred_lj_ra - gt_lj, axis=-1).mean())
        rh_mpjpe = float(np.linalg.norm(pred_rj_ra - gt_rj, axis=-1).mean())
        sum_lh_mpjpe += lh_mpjpe
        sum_rh_mpjpe += rh_mpjpe
        sum_hand_mpjpe += (lh_mpjpe + rh_mpjpe) / 2.0

        # PA alignment per hand (match rigid_align behavior)
        pred_lj_pa = compute_similarity_transform(pred_lj_ra, gt_lj)
        pred_rj_pa = compute_similarity_transform(pred_rj_ra, gt_rj)
        lh_pa = float(np.linalg.norm(pred_lj_pa - gt_lj, axis=-1).mean())
        rh_pa = float(np.linalg.norm(pred_rj_pa - gt_rj, axis=-1).mean())
        sum_lh_pa_mpjpe += lh_pa
        sum_rh_pa_mpjpe += rh_pa
        sum_hand_pa_mpjpe += (lh_pa + rh_pa) / 2.0

        used += 1

    if used == 0:
        raise RuntimeError("No samples evaluated (all missing?). Check paths/format.")

    # Convert to mm (already mean distance in meters -> mm)
    mpjpe_ra_mm = (sum_mpjpe_ra / used) * 1000.0
    mpvpe_ra_mm = (sum_mpvpe_ra / used) * 1000.0
    pa_mpjpe_mm = (sum_pa_mpjpe / used) * 1000.0
    pa_mpvpe_mm = (sum_pa_mpvpe / used) * 1000.0

    lh_mpvpe_mm   = (sum_lh_mpvpe_ra / used) * 1000.0
    rh_mpvpe_mm   = (sum_rh_mpvpe_ra / used) * 1000.0
    hand_mpvpe_mm = (sum_hand_mpvpe_ra / used) * 1000.0

    lh_pa_mpvpe_mm   = (sum_lh_pa_mpvpe / used) * 1000.0
    rh_pa_mpvpe_mm   = (sum_rh_pa_mpvpe / used) * 1000.0
    hand_pa_mpvpe_mm = (sum_hand_pa_mpvpe / used) * 1000.0

    face_mpvpe_mm    = (sum_face_mpvpe_ra / used) * 1000.0
    face_pa_mpvpe_mm = (sum_face_pa_mpvpe / used) * 1000.0
    lh_mpjpe_mm = (sum_lh_mpjpe / used) * 1000.0
    rh_mpjpe_mm = (sum_rh_mpjpe / used) * 1000.0
    hand_mpjpe_mm = (sum_hand_mpjpe / used) * 1000.0
    lh_pa_mpjpe_mm = (sum_lh_pa_mpjpe / used) * 1000.0
    rh_pa_mpjpe_mm = (sum_rh_pa_mpjpe / used) * 1000.0
    hand_pa_mpjpe_mm = (sum_hand_pa_mpjpe / used) * 1000.0


    print("============================================================")
    print(f"Evaluated: {used} samples (missing: {missed}, skip_missing={args.skip_missing})")
    print("---- Full body ----")
    print(f"MPJPE (pelvis-align, mm)   : {mpjpe_ra_mm:.3f}")
    print(f"PA-MPJPE (mm)              : {pa_mpjpe_mm:.3f}")
    print(f"MPVPE (pelvis-align, mm)   : {mpvpe_ra_mm:.3f}")
    print(f"PA-MPVPE (mm)              : {pa_mpvpe_mm:.3f}")
    print("---- Hands (vertex, MANO_SMPLX_vertex_ids) ----")
    print(f"L-hand MPVPE (wrist-align) : {lh_mpvpe_mm:.3f}")
    print(f"R-hand MPVPE (wrist-align) : {rh_mpvpe_mm:.3f}")
    print(f"Hands  MPVPE (avg L/R)     : {hand_mpvpe_mm:.3f}")
    print(f"L-hand PA-MPVPE            : {lh_pa_mpvpe_mm:.3f}")
    print(f"R-hand PA-MPVPE            : {rh_pa_mpvpe_mm:.3f}")
    print(f"Hands  PA-MPVPE (avg L/R)  : {hand_pa_mpvpe_mm:.3f}")
    print("---- Hand joints (21j, wrist-align) ----")
    print(f"L hand MPJPE             : {lh_mpjpe_mm:.3f}")
    print(f"R hand MPJPE             : {rh_mpjpe_mm:.3f}")
    print(f"Hand  MPJPE (avg)        : {hand_mpjpe_mm:.3f}")
    print(f"L hand PA-MPJPE          : {lh_pa_mpjpe_mm:.3f}")
    print(f"R hand PA-MPJPE          : {rh_pa_mpjpe_mm:.3f}")
    print(f"Hand  PA-MPJPE (avg)     : {hand_pa_mpjpe_mm:.3f}")
    print("---- Face (vertex, FLAME ids) ----")
    print(f"Face  MPVPE (neck-align)   : {face_mpvpe_mm:.3f}")
    print(f"Face  PA-MPVPE             : {face_pa_mpvpe_mm:.3f}")
    print("============================================================")


if __name__ == "__main__":
    main()