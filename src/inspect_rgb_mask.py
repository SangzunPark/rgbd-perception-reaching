
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from object_perception import (
    detect_red_object,
    select_mask_center_pixel,
)

# 1. Locate project outputs
ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"

# 2. Load RGB image
image = plt.imread(OUTPUT_DIR / "rgb.png")
rgb = image[:, :, :3]

# 3. Detect red pixels
mask = detect_red_object(rgb)
# Select a representative pixel automatically
u, v = select_mask_center_pixel(mask)

# 4. Save binary mask
plt.imsave(
    OUTPUT_DIR / "red_mask.png",
    mask.astype(float), # 기존 bool 값을 1과 0으로 변환
    cmap="gray",
    vmin=0,
    vmax=1,
)

# 5. Visualize RGB and mask side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(rgb)
axes[0].scatter(
    u, v,
    color="yellow",
    marker="+",
    s=100,
    linewidths=2,
)
axes[0].set_title("Original RGB Image")
axes[0].axis("off")

axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Detected Red Mask")
axes[1].axis("off")

plt.tight_layout()

fig.savefig(
    OUTPUT_DIR / "rgb_mask_comparison.png",
    dpi=150,
)

plt.close(fig)

# 6. Print results
print("RGB shape:", rgb.shape)
print("Mask shape:", mask.shape)
print("Detected pixels:", np.count_nonzero(mask))
print("Saved: outputs/red_mask.png")
print("Saved: outputs/rgb_mask_comparison.png")
print(f"Selected pixel (u, v): ({u}, {v})")
print("Selected pixel is in mask:", mask[v, u])