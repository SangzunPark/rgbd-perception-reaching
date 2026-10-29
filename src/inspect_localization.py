
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.camera_geometry import compute_intrinsics
from src.object_perception import (
    detect_red_object,
    select_mask_center_pixel,
)
from src.object_localization import estimate_target


# 1. Load data
ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"

rgb = plt.imread(OUTPUT_DIR / "rgb.png")[:, :, :3]
depth_m = np.load(OUTPUT_DIR / "depth_m.npy")
view = np.load(OUTPUT_DIR / "view_matrix.npy")
gt_top = np.load(OUTPUT_DIR / "gt_top_position.npy")

height, width = depth_m.shape
K = compute_intrinsics(width, height, 60)

# 2. Detect object pixel
mask = detect_red_object(rgb)
u, v = select_mask_center_pixel(mask)

# 3. Estimate with both methods
# single은 한칸만, median은 5x5의 평균
p_single = estimate_target(
    rgb, depth_m, K, view,
    depth_method="single",
)

p_median = estimate_target(
    rgb, depth_m, K, view,
    depth_method="median",
)

# 4. Compute XYZ differences (in millimeters)
diff_single = (p_single - gt_top) * 1000
diff_median = (p_median - gt_top) * 1000

# 5. Compute Euclidean errors
error_single = np.linalg.norm(diff_single)
error_median = np.linalg.norm(diff_median)

# 6. Print evaluation
print("Selected pixel:", (u, v))
print("GT top World:", np.round(gt_top, 6))

print("\n--- Single Depth ---")
print("Estimated:", np.round(p_single, 6))
print("XYZ error (mm):", np.round(diff_single, 3))
print(f"3D error: {error_single:.3f} mm")

print("\n--- Median Depth ---")
print("Estimated:", np.round(p_median, 6))
print("XYZ error (mm):", np.round(diff_median, 3))
print(f"3D error: {error_median:.3f} mm")

# 7. Visualize detection and World XY positions
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].imshow(rgb)
axes[0].scatter(
    u, v, color="yellow", marker="+",
    s=120, linewidths=2,
)
axes[0].set_title("Detected Target Pixel")
axes[0].axis("off")

axes[1].scatter(
    gt_top[0], gt_top[1],
    color="green", marker="*", s=140,
    label="GT top",
)

axes[1].scatter(
    p_single[0], p_single[1],
    facecolors="none", edgecolors="blue",
    s=140, linewidths=2, label="Single",
)

axes[1].scatter(
    p_median[0], p_median[1],
    color="red", marker="x", s=65,
    label="Median",
)

axes[1].set_xlabel("World X (m)")
axes[1].set_ylabel("World Y (m)")
axes[1].set_title("World XY Localization")
axes[1].set_aspect("equal", adjustable="datalim")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
fig.savefig(
    OUTPUT_DIR / "localization_comparison.png",
    dpi=150,
)
plt.close(fig)

print("\nSaved: outputs/localization_comparison.png")
