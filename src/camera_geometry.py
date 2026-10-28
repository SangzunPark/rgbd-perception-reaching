
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


# OpenGL Frame
def optical_to_opengl(point_optical):
    """Convert a point from optical to OpenGL camera frame."""

    # 1. Convert input to NumPy array
    p = np.asarray(point_optical, dtype=float)

    if p.shape != (3,):
        raise ValueError("Point must have shape (3,)")

    # 2. Rotation matrix: Optical -> OpenGL
    R_gl_from_optical = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1],
    ])

    # 3. Transform coordinates, @ means Numpy matrix multiplicaion 
    point_opengl = R_gl_from_optical @ p

    return point_opengl


# World P
def opengl_to_world(point_opengl, view_matrix):
    """Convert OpenGL camera coordinates to World coordinates."""

    # 1. Convert point to NumPy array
    p = np.asarray(point_opengl, dtype=float)

    if p.shape != (3,):
        raise ValueError("Point must have shape (3,)")

    # 2. Convert PyBullet view matrix to 4x4 / 동차행렬: 곱셈으로 덧셈(이동)을 구현
    # OpenGL은 컬럼메이저 형식으로 저장하기 때문에 올바른 해석을 위해 order =F 값을 지정 
    V = np.asarray(view_matrix, dtype=float).reshape(
        (4, 4), order="F"
    )

    # 3. Inverse: Camera -> World, 역행렬 계산을 위한 Numpy 함수
    T_world_from_gl = np.linalg.inv(V)

    # 4. Convert 3D point to homogeneous coordinates
    p_h = np.append(p, 1.0)

    # 5. Transform point to World coordinates
    p_world_h = T_world_from_gl @ p_h

    # 6. Return XYZ
    return p_world_h[:3]
