
import numpy as np


def detect_red_object(rgb):
    """Detect red pixels in an RGB image normalized to [0, 1]."""

    # 1. Convert to NumPy array
    rgb = np.asarray(rgb)

    # 2. Check RGB image shape
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB image must have shape (H, W, 3)")

    # 3. Extract RGB channels
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    # 4. Apply red color thresholds
    red_mask = (
        (r > 0.6)
        & (r > 1.5 * g)
        & (r > 1.5 * b)
    )

    # 5. Return binary mask
    return red_mask


def select_mask_center_pixel(mask):
    """Select a detected pixel closest to the mask centroid."""

    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D array")

    # 1. Get coordinates of detected pixels
    rows, cols = np.where(mask)

    # 2. Check detection failure
    if len(rows) == 0:
        raise ValueError("No object pixels detected")

    # 3. Compute mask centroid
    center_u = cols.mean()
    center_v = rows.mean()

    # 4. Find the detected pixel nearest to centroid
    distances_sq = (
        (cols - center_u) ** 2
        + (rows - center_v) ** 2
    )

    index = np.argmin(distances_sq)

    # 5. Return actual pixel coordinates (u, v)
    u = int(cols[index])
    v = int(rows[index])

    return u, v
