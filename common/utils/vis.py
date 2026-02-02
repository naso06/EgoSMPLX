import os
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender
import trimesh
from config import cfg
from FishEyeCalibrated import FishEyeCameraCalibrated
import torch
torch.cuda.empty_cache()


def vis_keypoints_with_skeleton(img, kps, kps_lines, kp_thresh=0.4, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_keypoints(img, kps, alpha=1, radius=3, color=None):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    if color is None:
        colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
        colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for i in range(len(kps)):
        p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
        if color is None:
            cv2.circle(kp_mask, p, radius=radius, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(kp_mask, p, radius=radius, color=color, thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_mesh(img, mesh_vertex, alpha=0.5):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(mesh_vertex))]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    mask = np.copy(img)

    # Draw the mesh
    for i in range(len(mesh_vertex)):
        p = mesh_vertex[i][0].astype(np.int32), mesh_vertex[i][1].astype(np.int32)
        cv2.circle(mask, p, radius=1, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, mask, alpha, 0)

def vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d[i1,0], kpt_3d[i2,0]])
        y = np.array([kpt_3d[i1,1], kpt_3d[i2,1]])
        z = np.array([kpt_3d[i1,2], kpt_3d[i2,2]])

        if kpt_3d_vis[i1,0] > 0 and kpt_3d_vis[i2,0] > 0:
            ax.plot(x, z, -y, c=colors[l], linewidth=2)
        if kpt_3d_vis[i1,0] > 0:
            ax.scatter(kpt_3d[i1,0], kpt_3d[i1,2], -kpt_3d[i1,1], c=colors[l], marker='o')
        if kpt_3d_vis[i2,0] > 0:
            ax.scatter(kpt_3d[i2,0], kpt_3d[i2,2], -kpt_3d[i2,1], c=colors[l], marker='o')

    x_r = np.array([0, cfg.input_shape[1]], dtype=np.float32)
    y_r = np.array([0, cfg.input_shape[0]], dtype=np.float32)
    z_r = np.array([0, 1], dtype=np.float32)
    
    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.legend()

    plt.show()
    cv2.waitKey(0)

def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()


def perspective_projection(vertices, cam_param):
    # vertices: [N, 3]
    # cam_param: [3]
    fx, fy= cam_param['focal']
    cx, cy = cam_param['princpt']
    vertices[:, 0] = vertices[:, 0] * fx / vertices[:, 2] + cx
    vertices[:, 1] = vertices[:, 1] * fy / vertices[:, 2] + cy
    return vertices


def render_mesh(img, mesh, face, cam_param, mesh_as_vertices=False):
    if mesh_as_vertices:
        # to run on cluster where headless pyrender is not supported for A100/V100
        vertices_2d = perspective_projection(mesh, cam_param)
        img = vis_keypoints(img, vertices_2d, alpha=0.8, radius=2, color=(0, 0, 255))
    else:
        # mesh
        mesh = trimesh.Trimesh(mesh, face)
        rot = trimesh.transformations.rotation_matrix(
        np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
        scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        focal, princpt = cam_param['focal'], cam_param['princpt']
        camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
        scene.add(camera)

        # renderer
        renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)

        # light
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        # render
        rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        rgb = rgb[:,:,:3].astype(np.float32)
        valid_mask = (depth > 0)[:,:,None]

        # save to image
        img = rgb * valid_mask + img * (1-valid_mask)

    return img


def render_mesh_fisheye(
    img: np.ndarray,
    mesh: np.ndarray,
    face: np.ndarray,
    cam_param: dict,
    fisheye_cam: FishEyeCameraCalibrated,
    mesh_as_vertices: bool = False,   # backward compat
    mode: str = "soild",          # "points" | "wireframe" | "solid"
    mesh_color=(255, 255, 255),       # BGR
    alpha: float = 0.5
) -> np.ndarray:
    """
    fisheye 왜곡을 반영해 SMPL mesh를 렌더링합니다.
    mode:
      - "points":   버텍스 점 렌더
      - "wireframe":삼각형 모서리만 렌더
      - "solid":    면 채워서 렌더(간단 z-정렬)
    """
    # 3D 버텍스 준비
    mesh_np = mesh if isinstance(mesh, np.ndarray) else mesh.detach().cpu().numpy()

    # 3D→2D fisheye 투영: proj2d (N,2)
    proj2d = fisheye_cam.world2camera(mesh_np)   # Nx2 (픽셀)
    # 깊이: 카메라 좌표계 z 사용 (정렬용)
    # world2camera가 2D만 주면, 원래 카메라좌표(왜곡 적용 전)의 z를 쓰면 충분
    z = mesh_np[:, 2].copy()  # (N,)

    # 안전 처리
    H, W = img.shape[:2]
    overlay = img.copy()

    # 레거시 스위치 호환
    if mesh_as_vertices and mode == "wireframe":
        mode = "points"
    if mesh_as_vertices and mode == "solid":
        mode = "points"

    if mode == "points":
        return vis_keypoints(img, proj2d, alpha=0.8, radius=2, color=(0, 0, 255))

    if mode == "wireframe":
        for tri in face:
            pts = proj2d[tri].astype(np.int32)
            cv2.polylines(overlay, [pts], isClosed=True, color=mesh_color, thickness=1, lineType=cv2.LINE_AA)
        return cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

    if mode == "solid":
        # 간단한 z-정렬: 삼각형 중심 z(작을수록 카메라 가까움) 기준으로 뒤→앞 순서로 그림
        # 주의: 실제 z-buffer가 아니라 평균 z 정렬이라 일부 교차에서 부정확할 수 있음
        tris = face.astype(np.int32)
        tri_depth = z[tris].mean(axis=1)  # (F,)
        order = np.argsort(tri_depth)[::-1]  # far->near (깊은 것부터 그려서 가려지게)

        fill_color = tuple(int(c) for c in mesh_color)  # BGR
        for idx in order:
            pts = proj2d[tris[idx]].astype(np.int32)

            # 화면 경계밖 NaN/Inf 방지
            if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
                continue

            # 폴리곤 채우기
            cv2.fillConvexPoly(overlay, pts, color=fill_color, lineType=cv2.LINE_AA)

        return cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

    # 알 수 없는 모드면 wireframe으로
    for tri in face:
        pts = proj2d[tri].astype(np.int32)
        cv2.polylines(overlay, [pts], True, mesh_color, 1, cv2.LINE_AA)
    return cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

