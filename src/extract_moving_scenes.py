"""
extract_moving_scenes.py
------------------------
Extract only 'moving' scenes (mean ego speed > threshold)
from nuScenes-mini preprocessing results.
Copies RGB + BEV + IMU triplets for these scenes into:
data/preprocessed_moving/
"""

import os, h5py, numpy as np
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from src.dataset_utils import save_triplet


# ---------------------------------------------------------------------
# Compute mean ego speed per scene
# ---------------------------------------------------------------------
def compute_mean_speed(nusc, scene_name):
    """Compute mean ego speed for one scene."""
    scene = [s for s in nusc.scene if s["name"] == scene_name][0]
    token = scene["first_sample_token"]
    poses, times = [], []

    while token:
        sample = nusc.get("sample", token)
        sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        pose = nusc.get("ego_pose", sd["ego_pose_token"])
        poses.append(np.array(pose["translation"]))
        times.append(pose["timestamp"] / 1e6)
        token = sample["next"]

    poses, times = np.stack(poses), np.array(times)
    v = np.diff(poses, axis=0) / np.diff(times)[:, None]
    return np.mean(np.linalg.norm(v, axis=1))


# ---------------------------------------------------------------------
# Extract and copy moving-scene triplets
# ---------------------------------------------------------------------
def extract_moving_scenes(
    nusc,
    src_root="data/preprocessed",
    out_root="data/preprocessed_moving",
    speed_threshold=0.5,
):
    """Copy samples from moving scenes into a new folder."""
    os.makedirs(out_root, exist_ok=True)

    print(" Scanning scenes for motion ...")
    moving_scenes = [
        s["name"]
        for s in nusc.scene
        if compute_mean_speed(nusc, s["name"]) > speed_threshold
    ]
    print(f" Found {len(moving_scenes)} moving scenes: {moving_scenes}\n")

    # Copy sequentially numbered .h5 files by sample index
    print(" Copying moving samples ...")
    count = 0

    for s_name in tqdm(moving_scenes):
        scene = [s for s in nusc.scene if s["name"] == s_name][0]
        token = scene["first_sample_token"]

        while token:
            sample = nusc.get("sample", token)
            idx = nusc.sample.index(sample)
            fname = f"sample_{idx:05d}.h5"
            src_path = os.path.join(src_root, fname)
            dest_path = os.path.join(out_root, fname)

            if os.path.exists(src_path):
                with h5py.File(src_path, "r") as f_in, h5py.File(dest_path, "w") as f_out:
                    for key in f_in.keys():
                        f_out.create_dataset(key, data=f_in[key][()])
                count += 1

            token = sample["next"]

    print(f" Copied {count} moving samples → {out_root}")
