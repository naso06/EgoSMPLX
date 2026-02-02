import os
import subprocess
from glob import glob
from mmhuman3d.data.data_structures.human_data import HumanData
import numpy as np
import json
from tqdm import tqdm
os.environ["PYOPENGL_PLATFORM"] = "osmesa"


def run_inference_on_scene(scene_path, output_path, smplerx_infer_script):
    external_imgs_path = os.path.join(scene_path, 'external_imgs')
    cmd = [
        'python', smplerx_infer_script,
        '--img_path', external_imgs_path,
        '--output_folder', output_path,
        '--pretrained_model', 'smpler_x_h32',
        '--num_gpus', '1',
        '--save_mesh'
    ]
    subprocess.run(cmd, check=True)

def build_humandata(output_path, save_path):
    human_data = HumanData()
    smplx_dir = os.path.join(output_path, 'smplx')
    meta_dir = os.path.join(output_path, 'meta')

    image_paths = []
    bbox_xywh = []
    smplx_dict = {
        'global_orient': [],
        'body_pose': [],
        'left_hand_pose': [],
        'right_hand_pose': [],
        'jaw_pose': [],
        'leye_pose': [],
        'reye_pose': [],
        'betas': [],
        'expression': [],
        # 'transl': []
    }

    npz_files = sorted(glob(os.path.join(smplx_dir, '*.npz')))
    for npz_path in tqdm(npz_files):
        file_name = os.path.basename(npz_path).replace('.npz', '')
        meta_path = os.path.join(meta_dir, f'{file_name}.json')
        img_name = f'img/imgs_{file_name.split("_")[0]:06d}.jpg'

        if not os.path.exists(meta_path):
            continue

        image_paths.append(img_name)

        with open(meta_path) as f:
            meta = json.load(f)
            bbox_xywh.append(meta['bbox'] + [1.0])

        data = np.load(npz_path)
        for k in smplx_dict:
            smplx_dict[k].append(data[k])

    for k in smplx_dict:
        smplx_dict[k] = np.array(smplx_dict[k])

    human_data['image_path'] = image_paths
    human_data['bbox_xywh'] = np.array(bbox_xywh, dtype=np.float32)
    human_data['smplx'] = smplx_dict
    human_data.dump(os.path.join(save_path, 'human_data.npz'))

def process_egopw_all(base_dir, smplerx_infer_script):
    subjects = sorted(os.listdir(base_dir))
    for subject in subjects:
        subject_path = os.path.join(base_dir, subject)
        if not os.path.isdir(subject_path):
            continue

        scenes = sorted(os.listdir(subject_path))
        for scene in scenes:
            scene_path = os.path.join(subject_path, scene)
            if not os.path.isdir(scene_path):
                continue

            print(f"📌 Processing: {subject}/{scene}")
            output_path = scene_path  # 저장 경로 동일
            run_inference_on_scene(scene_path, output_path, smplerx_infer_script)
            build_humandata(output_path, output_path)

if __name__ == '__main__':
    egopw_root = '/media/cv1/SeagateHDD2TB/EgoPW'
    smplerx_infer_script = '/home/cv1/works/SMPLer-X/main/inference.py'
    process_egopw_all(egopw_root, smplerx_infer_script)