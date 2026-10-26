
from pathlib import Path

import numpy as np

from camera_geometry import (
    compute_intrinsics,
    pixel_to_camera,
)


# 1. Locate depth file
ROOT_DIR = Path(__file__).resolve().parent.parent
depth_path = ROOT_DIR / "outputs" / "depth_m.npy"

# 2. Load metric depth
depth_m = np.load(depth_path)

print("Depth shape:", depth_m.shape)

# 3. Camera intrinsics
height, width = depth_m.shape
K = compute_intrinsics(width, height, 60)

print("Camera intrinsics K:")
print(K)

# 4. Select pixels (u, v)
pixels = [
    (160, 120),
    (182, 109),
]

# 5. Convert each pixel to camera 3D
for u, v in pixels:
    z = depth_m[v, u]

    point_camera = pixel_to_camera(u, v, z, K)

    print(f"\nPixel: ({u}, {v})")
    print(f"Depth: {z:.4f} m")
    print("Camera 3D:", np.round(point_camera, 4))
