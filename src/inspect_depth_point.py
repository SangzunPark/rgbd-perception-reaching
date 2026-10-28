from pathlib import Path

import numpy as np

from camera_geometry import (
    compute_intrinsics,
    pixel_to_camera,
    optical_to_opengl,
    opengl_to_world,
)


# 1. Locate output files
ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"

# 2. Load the actual rendered data
depth_m = np.load(OUTPUT_DIR / "depth_m.npy")
view = np.load(OUTPUT_DIR / "view_matrix.npy")

print("Depth shape:", depth_m.shape)
print("View matrix shape:", view.shape)

# 3. Camera intrinsics
height, width = depth_m.shape
K = compute_intrinsics(width, height, 60)

# 4. Select pixels
pixels = [
    (160, 120),  # Floor
    (182, 109),  # Cube top surface
]

# 5. Complete coordinate transformation
for u, v in pixels:

    # Step A: Read depth at pixel
    z = depth_m[v, u] # NumPy order: [row, column] = [v, u]

    # Step B: Pixel -> Optical Camera 3D
    p_optical = pixel_to_camera(u, v, z, K)

    # Step C: Optical -> OpenGL Camera 3D
    p_opengl = optical_to_opengl(p_optical)

    # Step D: OpenGL Camera -> World 3D
    p_world = opengl_to_world(p_opengl, view)

    # Step E: Print results
    print(f"\nPixel: ({u}, {v})")
    print(f"Depth: {z:.4f} m")
    print("Optical:", np.round(p_optical, 4))
    print("OpenGL: ", np.round(p_opengl, 4))
    print("World:  ", np.round(p_world, 4))

    
    # 6. Compare the selected cube pixel with GT top center
    if (u, v) == (182, 109):

        # Load cube center ground truth
        gt_center = np.load(
            OUTPUT_DIR / "gt_position.npy"
        )

        # Approximate top-face center (axis-aligned cube)
        cube_half_height = 0.025

        gt_top = gt_center + np.array([
            0.0,
            0.0,
            cube_half_height,
        ])

        # Difference between estimated point and GT target
        difference = p_world - gt_top

        # Euclidean distance
        error_m = np.linalg.norm(difference)

        print("\n--- GT Comparison ---")
        print("GT top:", np.round(gt_top, 6))
        print("Estimated:", np.round(p_world, 6))
        print("XYZ difference (m):", np.round(difference, 6))
        print(f"Distance to GT top: {error_m * 1000:.3f} mm")
    

