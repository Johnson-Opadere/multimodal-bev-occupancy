import cv2
import numpy as np
import os

def preprocess_rgb(img_path, size=(256, 256), normalize=True):
    """
    Load and normalize an RGB image for model input.
    - Converts BGR → RGB
    - Resizes
    - Scales to [0,1] or [-1,1]
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    if normalize:
        img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
    img = np.transpose(img, (2, 0, 1))  # CHW
    return img

def lidar_to_bev(points, x_range=(-50,50), y_range=(-50,50), res=0.5):
    """
    Convert raw LiDAR point cloud (x, y, z) to a binary BEV occupancy grid.
    """
    x, y = points[:,0], points[:,1]
    mask = (x>x_range[0]) & (x<x_range[1]) & (y>y_range[0]) & (y<y_range[1])
    x, y = x[mask], y[mask]
    bev_w = int((x_range[1]-x_range[0])/res)
    bev_h = int((y_range[1]-y_range[0])/res)
    bev = np.zeros((bev_h, bev_w), np.uint8)
    xi = ((x - x_range[0])/res).astype(int)
    yi = ((y - y_range[0])/res).astype(int)
    bev[bev_h-1-yi, xi] = 1
    return bev
