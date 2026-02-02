import os
import os.path as osp
import numpy as np
from pathlib import Path

def extract_subject_id(p: str) -> str:
    for part in Path(p).parts:
        if part.startswith("render_people_"):
            return part[len("render_people_"):]
    return Path(p).parent.name

def save_subsampled_image_paths_txt(
    npz_path: str,
    out_txt: str,
    img_root: str = "/media/cv1/SeagateHDD2TB/EgoWholeMocap",
    old_root: str = "/data2/EgoWholeMocap",
    sample_interval: int = 100,
):
    data = np.load(npz_path, allow_pickle=True)
    image_paths = data["image_path"]

    abs_image_paths = []
    for ip in image_paths:
        ip = str(ip).replace("\\", "/")
        if osp.isabs(ip):
            if ip.startswith(old_root):
                ip = ip.replace(old_root, img_root, 1)
            abs_image_paths.append(ip)
        else:
            abs_image_paths.append(osp.join(img_root, ip))

    # (선택) id 추출이 필요하면 유지
    _ = [extract_subject_id(p) for p in abs_image_paths]

    N = len(abs_image_paths)
    idxs = list(range(0, N, int(sample_interval)))

    with open(out_txt, "w") as f:
        for i in idxs:
            f.write(abs_image_paths[i] + "\n")

    print(f"[OK] Saved {len(idxs)} paths to: {out_txt}")
    print(f"     Total images: {N}, interval={sample_interval}")

if __name__ == "__main__":
    npz_path = "/media/cv1/SeagateHDD2TB/EgoWholeMocap/human_data_train/humandata.npz"
    out_txt  = "/media/cv1/SeagateHDD2TB/EgoWholeMocap/human_data_train/subsample_interval100_paths.txt"
    save_subsampled_image_paths_txt(npz_path, out_txt, sample_interval=100)
