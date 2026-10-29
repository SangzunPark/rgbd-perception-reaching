import numpy as np

from src.object_perception import (
    detect_red_object,
    select_mask_center_pixel,
)

from src.camera_geometry import (
    pixel_to_camera,
    optical_to_opengl,
    opengl_to_world,
)


def estimate_target(
    rgb, depth_m, K, view_matrix,
    depth_method="single",
    radius=2,
):
    """Estimate a red object's surface point in World coordinates."""

    # 1. Check RGB and Depth image sizes
    if depth_m.shape != rgb.shape[:2]:
        raise ValueError("RGB and Depth image sizes do not match")

    # 2. Detect the red object using RGB
    mask = detect_red_object(rgb)

    # 3. Select a representative pixel
    u, v = select_mask_center_pixel(mask)

    # 4. Select depth estimation method
    if depth_method == "single":
        z = depth_m[v, u]

    elif depth_method == "median":
        z = local_median_depth(
            depth_m, mask, u, v, radius=radius
        )

    else:
        raise ValueError(
            "depth_method must be 'single' or 'median'"
        )

    # 5. Pixel -> Optical Camera 3D
    p_optical = pixel_to_camera(u, v, z, K)

    # 6. Optical -> OpenGL Camera 3D
    p_opengl = optical_to_opengl(p_optical)

    # 7. OpenGL Camera -> World 3D
    p_world = opengl_to_world(p_opengl, view_matrix)

    return p_world


# radious =2 란 선택한 픽셀에서 좌우, 상하로 각각 2픽셀까지 포함한다는 뜻
# 따라서 선택한 픽셀을 중심으로 최대 5x5 영역을 검사
def local_median_depth(depth_m, mask, u, v, radius=2):
    """Compute median depth within a local object-mask region."""

    # 1. Check window radius
    if not isinstance(radius, (int, np.integer)) or radius < 0:
        raise ValueError("Radius must be a non-negative integer")

    height, width = depth_m.shape

    # 2. Define local window around selected pixel
    y0 = max(0, v - radius)
    y1 = min(height, v + radius + 1)

    x0 = max(0, u - radius)
    x1 = min(width, u + radius + 1)

    # 3. Extract depth and mask patches
    depth_patch = depth_m[y0:y1, x0:x1]
    mask_patch = mask[y0:y1, x0:x1]

    # 4. Keep only valid object depths
    # 해당 영역만 잘라내기 depth와 mask = True 영역
    valid = (
        mask_patch
        & np.isfinite(depth_patch)
        & (depth_patch > 0)
    )

    values = depth_patch[valid]

    # 5. Explicit failure if no valid depths
    if values.size == 0:
        raise ValueError("No valid depth in local object mask")

    # 6. Return median metric depth
    return float(np.median(values))
