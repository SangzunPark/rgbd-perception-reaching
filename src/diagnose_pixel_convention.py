
"""Diagnose renderer pixel alignment across image resolutions."""

import numpy as np
import pybullet as p

from src.camera_geometry import compute_intrinsics


CUBE_HALF = 0.025

RESOLUTIONS = [
    (320, 240),
    (640, 480),
    (321, 241),
]


def run_experiment(width, height):
    """Compare rendered mask center with projected GT center."""

    client = p.connect(p.DIRECT)

    try:
        # 1. Fixed cube centered at World origin
        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[CUBE_HALF] * 3,
        )

        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[CUBE_HALF] * 3,
            rgbaColor=[1, 0, 0, 1],
        )

        # 큐브 생성, 질량을 이번에는 0으로 줘서 중력이나 시뮬레이션 영향도 제거
        cube_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[0, 0, CUBE_HALF],
        )

        # 2. Same camera pose and FOV
        view = p.computeViewMatrix(
            cameraEyePosition=[0, 0, 1.0],
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[0, 1, 0],
        )

        projection = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=width / height,
            nearVal=0.05,
            farVal=2.0,
        )

        # 3. Render segmentation
        image = p.getCameraImage(
            width,
            height,
            viewMatrix=view,
            projectionMatrix=projection,
            renderer=p.ER_TINY_RENDERER,
        )

        segmentation = np.asarray(image[4]).reshape(
            height, width
        )

        mask = segmentation == cube_id
        rows, cols = np.where(mask)

        if rows.size == 0:
            raise ValueError("Cube not visible")

        u_mask = float(cols.mean())
        v_mask = float(rows.mean())

        # 4. GT top-face center in homogeneous coordinates
        gt_world = np.array([
            0.0, 0.0, 2 * CUBE_HALF, 1.0
        ])

        V = np.asarray(view).reshape(4, 4, order="F")
        P = np.asarray(projection).reshape(4, 4, order="F")

        # 5. World -> Clip -> NDC
        clip = P @ V @ gt_world
        ndc = clip[:3] / clip[3]

        # 6. NDC -> image coordinates
        u_gt = (ndc[0] + 1) * width / 2
        v_gt = (1 - ndc[1]) * height / 2

        # 7. Verify against our intrinsic-based projection
        K = compute_intrinsics(width, height, 60)

        camera_gl = (V @ gt_world)[:3]

        u_k = (
            K[0, 0] * camera_gl[0] / (-camera_gl[2])
            + K[0, 2]
        )
        v_k = (
            K[1, 1] * (-camera_gl[1]) / (-camera_gl[2])
            + K[1, 2]
        )

        assert np.allclose(
            [u_gt, v_gt],
            [u_k, v_k],
            atol=1e-4,
        )

        # 8. Pixel-space differences
        du = u_mask - u_gt
        dv = v_mask - v_gt

        # 9. Equivalent metric displacement at cube top
        depth = -camera_gl[2]
        mm_per_pixel = depth / K[1, 1] * 1000

        return {
            "u_gt": u_gt,
            "v_gt": v_gt,
            "u_mask": u_mask,
            "v_mask": v_mask,
            "du": du,
            "dv": dv,
            "mm_per_pixel": mm_per_pixel,
            "y_offset_mm": -dv * mm_per_pixel,
        }

    finally:
        p.disconnect(client)


def main():
    print(
        "Resolution | GT (u,v)       | Mask (u,v)     "
        "| du    | dv    | Y offset"
    )
    print("-" * 88)

    for width, height in RESOLUTIONS:
        result = run_experiment(width, height)

        print(
            f"{width:3d}x{height:<3d}    | "
            f"({result['u_gt']:6.2f},{result['v_gt']:6.2f}) | "
            f"({result['u_mask']:6.2f},{result['v_mask']:6.2f}) | "
            f"{result['du']:+5.2f} | "
            f"{result['dv']:+5.2f} | "
            f"{result['y_offset_mm']:+7.3f} mm"
        )

    print("\nProjection consistency checks: PASSED")


if __name__ == "__main__":
    main()
