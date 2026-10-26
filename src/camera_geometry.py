
import numpy as np


def compute_intrinsics(width, height, vertical_fov_deg):
    """Compute the camera intrinsic matrix."""

    # 1. Convert degrees to radians
    fov_rad = np.deg2rad(vertical_fov_deg)

    # 2. Compute focal length in pixels
    fy = height / (2 * np.tan(fov_rad / 2))
    fx = fy

    # 3. Compute principal point
    cx = width / 2
    cy = height / 2

    # 4. Construct intrinsic matrix
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1],
    ])

    return K



def pixel_to_camera(u, v, depth_m, K):
    """Convert a pixel and its depth to a camera 3D point."""

    # 1. Get metric depth
    z = float(depth_m)

    if not np.isfinite(z) or z <= 0:
        raise ValueError("Invalid depth")

    # 2. Extract camera intrinsics
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # 3. Back-project pixel to 3D
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # 4. Return camera-frame coordinates
    return np.array([x, y, z])
