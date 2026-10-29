"""Week 0: PyBullet 장면을 만들고 RGB / Depth / Segmentation / GT 좌표를 저장한다."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # WSL에는 화면이 없으므로 파일 저장 전용 백엔드 사용
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p # 시뮬레이터
import pybullet_data

# 결과를 저장할 폴더: 프로젝트 루트의 outputs/
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

CUBE_HALF = 0.025                    # 큐브 한 변 5cm → 절반 2.5cm
CUBE_START = [0.10, 0.05, CUBE_HALF]  # 바닥 위에 놓인 위치 (World 좌표, m)

client = p.connect(p.DIRECT)  # 창 없이 실행하는 모드
try:
    # 1. 기본 환경 설정
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81) # -9.81은 지구 중력 크기 

    # 2. 바닥 생성
    p.loadURDF("plane.urdf")

    # 3. 빨간 큐브 생성
    # visual 만 있으면 물체간 서로 통과 가능하기 때문에 collisionShape 값을 생성
    # 이 조합이 합쳐져 바닥에 큐브가 있는 환경 구현
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[CUBE_HALF] * 3)
    vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[CUBE_HALF] * 3, rgbaColor=[1, 0, 0, 1]
    )
    cube_id = p.createMultiBody(
        baseMass=0.1, # 질량이 0이면 고정된 물체 취급이라 어느정도 중략을 준다
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=CUBE_START,
    )

    # 물체가 바닥에 자리 잡도록 잠시 시뮬레이션 진행
    # pybullet의 1초는 240
    for _ in range(240):
        p.stepSimulation()

    # Ground Truth: 시뮬레이터가 알고 있는 실제 큐브 중심 위치
    # gt_pos는 큐브 중심의 world 좌표
    # gt_orn은 큐브의 회전 방향을 나타내는 quaternion
    gt_pos, gt_orn = p.getBasePositionAndOrientation(cube_id)

    # 4. 위에서 내려다보는 가상 카메라 설정
    width, height = 320, 240
    near, far = 0.05, 2.0
    camera_height = 1.0

    # 카메라 위치 지정, 어디에 서서 어느 쪽을 볼까?
    view = p.computeViewMatrix(
        cameraEyePosition=[0, 0, camera_height], # eye = 카메라위치
        cameraTargetPosition=[0, 0, 0], # target = 카메라가 바라보는점
        cameraUpVector=[0, 1, 0], # 카메라를 돌릴 수 있기 때문에 기준 방향 지정(위쪽)
    )
    # 어떤 렌즈로, 어떤 모양의 사진을 찍을까? Intrinsic (f_x, f_y, c_x, c_y)
    projection = p.computeProjectionMatrixFOV(
        fov=60, # 시야각(Field of View), fov는 intrinsic의 f 값을 결정
        aspect=width / height, # 사진의 모양, 320,240 이니 4:3
        nearVal=near, # 이 값보다 가까운것은 찍지 않는다, 손전등 연상과 유사
        farVal=far, # 이 값보다 먼것은 찍지 않는다.  
    )

    # 5. RGB, depth buffer, segmentation 렌더링
    # renderer는 CPU로 설정, GPU는 ER_BULLET_HARDWARE_OPENGL
    image = p.getCameraImage(
        width,
        height,
        viewMatrix=view,
        projectionMatrix=projection,
        renderer=p.ER_TINY_RENDERER,
    )
    # image가 받는 인자는 가로 / 세로 / 컬러사진 / 깊이사진 / 이름표사진
    # rgb 코드는 Numpy 일열 배열을 행렬로 변환, 엑셀로 비유하면
    # 240의 행과 320의 열이 있고 각 칸에 숫자 4개가 있으며
    # 빨강, 초록, 파랑, 투명도 중에 마지막 투명도는 버린다.
    # 마지막 segmentaion(이름표)는 처음만든 바닥(0값)과 다음 만든 물체(1값)을 분류해준다
    rgb = np.asarray(image[2], dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
    depth_buffer = np.asarray(image[3]).reshape(height, width)
    segmentation = np.asarray(image[4]).reshape(height, width)

    # 6. depth buffer(0~1)를 실제 미터 단위 깊이로 변환
    # 깊이 depth_buffer에 있는 숫자는 실제 숫자가 아니고 0과 1로 되어 있으며
    # 0은 내가 설정한 최소거리(5cm) 1은 최대거리(2m) 가 된다
    # 하지만 이것은 자의 눈금처럼 각 구간이 동일하지 않은데,
    # 이것은 컴퓨터그래픽이 원래 가까운 것을 더 정밀하게 그리기 때문이다.

    depth_m = near * far / (far - (far - near) * depth_buffer)

    # 7. 결과 저장
    # depth.png는 사람이 보는 용도, 값을 밝기로 바꿔서 실제 미터 값이 사라짐
    # depth_m.npy는 계산에 쓰는 용도, 실수 미터 값이 그대로 보존
    plt.imsave(OUTPUT_DIR / "rgb.png", rgb)
    plt.imsave(OUTPUT_DIR / "depth.png", depth_m, cmap="gray")
    np.save(OUTPUT_DIR / "depth_m.npy", depth_m)
    np.save(
    OUTPUT_DIR / "view_matrix.npy",
    np.asarray(view, dtype=float),
)

    cube_mask = segmentation == cube_id
    plt.imsave(
        OUTPUT_DIR / "segmentation.png",
        cube_mask.astype(float),
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    np.save(OUTPUT_DIR / "gt_position.npy", np.array(gt_pos))

    # Ground Truth of the cube's top-face center
    R_world_from_body = np.asarray(
        p.getMatrixFromQuaternion(gt_orn)
    ).reshape(3, 3)

    gt_top = np.asarray(gt_pos) + (
        R_world_from_body
        @ np.array([0.0, 0.0, CUBE_HALF])
    )

    np.save(
        OUTPUT_DIR / "gt_top_position.npy",
        gt_top,
    )


    # 8. 결과 출력과 검증
    print("RGB shape:", rgb.shape)
    print("Depth shape:", depth_m.shape)
    print("Cube pixels:", np.count_nonzero(cube_mask))
    print(f"Depth range (m): {depth_m.min():.3f} ~ {depth_m.max():.3f}")
    print("GT cube center (world, m):", np.round(gt_pos, 4))

    # 큐브 윗면은 바닥에서 5cm 높으므로, 카메라와의 거리는 약 0.95m여야 한다
    expected_top = camera_height - 2 * CUBE_HALF
    measured_top = depth_m[cube_mask].mean()
    floor_depth = depth_m[~cube_mask].mean()
    print(f"Cube top depth: measured {measured_top:.3f} m / expected {expected_top:.3f} m")
    print(f"Floor depth:    measured {floor_depth:.3f} m / expected {camera_height:.3f} m")
finally:
    p.disconnect()