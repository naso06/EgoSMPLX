import cv2
import numpy as np
from FishEyeCalibrated import FishEyeCameraCalibrated
import os

# 입력 경로
img_path = "/media/cv1/SeagateHDD2TB/EgoWholeMocap/render_people_adanna/2hand Idle/img/000001.jpg"
calib_path = "/home/cv1/works/SMPLer-X/main/fisheye.calibration_05_08.json"

# 출력 크기 (언디스토션 이미지 크기)
undist_w, undist_h = 1280, 1024

# # 어안 카메라 보정 모델 초기화
# fisheye_camera = FishEyeCameraCalibrated(calib_path)


# # (1) 보정 이미지 기준 좌표 생성
# grid_x, grid_y = np.meshgrid(np.arange(undist_w), np.arange(undist_h))
# grid_2d = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)

# # (2) 단위 벡터 (기준 평면에 위치한 시선 방향 벡터)
# norm_x = (grid_2d[:, 0] - undist_w / 2) / (undist_w / 2)
# norm_y = (grid_2d[:, 1] - undist_h / 2) / (undist_h / 2)
# ray_3d = np.stack([norm_x, norm_y, np.ones_like(norm_x)], axis=-1)

# # (3) 유효한 3D 벡터 선택 (XY norm이 너무 작지 않은 경우만)
# norm_xy = np.linalg.norm(ray_3d[:, :2], axis=1)
# valid_mask = norm_xy > 1e-6

# # (4) 유효한 3D ray만 정규화하여 world2camera 적용
# ray_3d_valid = ray_3d[valid_mask]
# ray_3d_valid /= np.linalg.norm(ray_3d_valid, axis=1, keepdims=True)

# # (5) 결과 좌표 초기화 및 유효 영역만 변환
# fisheye_2d = np.full((ray_3d.shape[0], 2), -1, dtype=np.float32)
# for i, pt in zip(np.where(valid_mask)[0], ray_3d_valid):
#     try:
#         fisheye_2d[i] = fisheye_camera.world2camera(np.array([pt]))
#     except Exception:
#         fisheye_2d[i] = [-1, -1]  # fallback

# fisheye_2d = fisheye_2d.reshape((undist_h, undist_w, 2))

# # (6) 이미지 remap
fisheye_img = cv2.imread(img_path)
# undistorted_img = cv2.remap(
#     fisheye_img,
#     fisheye_2d[:, :, 0],
#     fisheye_2d[:, :, 1],
#     interpolation=cv2.INTER_LINEAR,
#     borderMode=cv2.BORDER_CONSTANT,
#     borderValue=0
# )

# cv2.imwrite("undistorted_fisheye.jpg", undistorted_img)

H_out, W_out = 1024, 1280
fov_deg = 120
fov_rad = np.deg2rad(fov_deg)

# Create output image grid
x = np.linspace(-1, 1, W_out)
y = np.linspace(-1, 1, H_out)
xx, yy = np.meshgrid(x, y)

# Define unit ray directions for each pixel
z = 1.0 / np.tan(fov_rad / 2)  # virtual focal length
rays = np.stack([xx, yy, np.full_like(xx, z)], axis=-1)
rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
rays_flat = rays.reshape(-1, 3)

# fisheye camera
fisheye_camera = FishEyeCameraCalibrated(calib_path)

# Convert 3D rays to 2D fisheye image pixels
mapped_pixels = []
for r in rays_flat:
    try:
        uv = fisheye_camera.world2camera(np.array([r]))  # (1, 2)
        mapped_pixels.append(uv[0])
    except:
        mapped_pixels.append([-1, -1])  # mark invalid

mapped_pixels = np.array(mapped_pixels, dtype=np.float32).reshape(H_out, W_out, 2)

# Remap
undistorted_wide = cv2.remap(
    fisheye_img,
    mapped_pixels[:, :, 0],
    mapped_pixels[:, :, 1],
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=0
)

cv2.imwrite("undistorted_wideview.jpg", undistorted_wide)