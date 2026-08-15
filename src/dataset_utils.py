"""
dataset_utils.py
----------------
Utility functions and dataset classes for loading and saving multimodal triplets:
- RGB (3×256×256)
- BEV occupancy (1×H×W)
- IMU sequence (optional, 10×6)
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# -------------------------------------------------------------------------
# SAVE TRIPLET UTILITY
# -------------------------------------------------------------------------
def save_triplet(out_path: str, rgb, bev, imu):
    """
    Save a preprocessed triplet (RGB, BEV, IMU) into an HDF5 file.

    Args:
        out_path (str): destination .h5 path
        rgb (np.ndarray or torch.Tensor): shape (3, H, W)
        bev (np.ndarray or torch.Tensor): shape (1, H, W)
        imu (np.ndarray): shape (T, 6) for time-series window
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Convert tensors to numpy if needed
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    if isinstance(bev, torch.Tensor):
        bev = bev.detach().cpu().numpy()

    with h5py.File(out_path, "w") as f:
        f.create_dataset("rgb", data=np.asarray(rgb, dtype=np.float32))
        f.create_dataset("bev", data=np.asarray(bev, dtype=np.float32))
        f.create_dataset("imu", data=np.asarray(imu, dtype=np.float32))


# -------------------------------------------------------------------------
# DATASET CLASS
# -------------------------------------------------------------------------
class RGBBEVDataset(Dataset):
    """
    Loads RGB–BEV pairs (and optional IMU) from preprocessed .h5 files.
    Each file contains:
        - rgb: [3×256×256]
        - bev: [1×H×W]  (typically 200×200 or 256×256)
        - imu: optional [10×6]

    Returns:
        rgb (torch.FloatTensor): [3×H×W]
        bev (torch.FloatTensor): [1×H×W]
        imu (torch.FloatTensor): [T×6]
    """

    def __init__(self, h5_paths):
        if not isinstance(h5_paths, list):
            raise TypeError("Expected a list of file paths for h5_paths.")
        self.h5_paths = h5_paths

    def __len__(self):
        return len(self.h5_paths)

    def __getitem__(self, idx):
        path = self.h5_paths[idx]
        if not os.path.exists(path):
            raise FileNotFoundError(f"[ERROR] Missing file: {path}")

        with h5py.File(path, "r") as f:
            rgb = np.array(f["rgb"], dtype=np.float32)
            bev = np.array(f["bev"], dtype=np.float32)
            imu = (
                np.array(f["imu"], dtype=np.float32)
                if "imu" in f
                else np.zeros((10, 6), np.float32)
            )

        # Convert numpy arrays → torch tensors
        rgb = torch.from_numpy(rgb)
        bev = torch.from_numpy(bev)
        imu = torch.from_numpy(imu)

        # Sanity: ensure consistent shape channels
        if rgb.ndim == 2:
            rgb = rgb.unsqueeze(0)
        if bev.ndim == 2:
            bev = bev.unsqueeze(0)

        return rgb, bev, imu
