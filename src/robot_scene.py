
"""Day 9-1: build an independent Panda + RGB-D scene in PyBullet."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data

from src.camera_geometry import compute_intrinsics
from src.object_perception import detect_red_object


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "robot_reaching"

# World coordinates in meters
ROBOT_BASE = (-0.50, 0.0, 0.0)
CUBE_XY = (0.10, 0.05)
CUBE_HALF = 0.025

# Panda's initial seven arm joint angles (radians)
HOME_ARM_Q = (
    0.0, -0.785, 0.0, -2.356,
    0.0, 1.571, 0.785,
)

ARM_NAMES = tuple(
    f"panda_joint{i}" for i in range(1, 8)
)
FINGER_NAMES = (
    "panda_finger_joint1",
    "panda_finger_joint2",
)
EE_LINK_NAME = "panda_grasptarget"

# Preserve the existing perception camera settings
WIDTH, HEIGHT, FOV = 320, 240, 60.0
NEAR, FAR = 0.05, 2.0


@dataclass(frozen=True)
class RobotScene:
    client_id: int
    plane_id: int
    cube_id: int
    robot_id: int
    arm_joint_indices: tuple[int, ...]
    finger_joint_indices: tuple[int, ...]
    ee_link_index: int


def create_robot_scene():
    """Create a Panda, red cube, and floor in one World."""

    client_id = p.connect(p.DIRECT) # No GUI Because of WSL

    if client_id < 0:
        raise RuntimeError("Could not connect to PyBullet")

    try:
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=client_id,
        )
        p.setGravity(
            0, 0, -9.81,
            physicsClientId=client_id,
        )
        p.setTimeStep(
            1.0 / 240.0,
            physicsClientId=client_id,
        )

        # 1. Floor
        plane_id = p.loadURDF(
            "plane.urdf",
            physicsClientId=client_id,
        )

        # 2. Red cube: same geometry as the old scenes
        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[CUBE_HALF] * 3,
            physicsClientId=client_id,
        )
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[CUBE_HALF] * 3,
            rgbaColor=[1, 0, 0, 1],
            physicsClientId=client_id,
        )
        cube_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[*CUBE_XY, CUBE_HALF],
            physicsClientId=client_id,
        )

        # Let the cube settle on the floor 240-> 1sec
        for _ in range(240):
            p.stepSimulation(
                physicsClientId=client_id
            )

        # 3. Load Panda after settling the cube
        # 로봇 등장 Unified Robot Description Format(URDF)
        robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=ROBOT_BASE,
            useFixedBase=True, # 바닥에 고정
            physicsClientId=client_id,
        )

        # 4. Discover joint and child-link indices

        # joint_indices["panda_joint4"] = 3,  "panda_joint4라는 이름은 3번"이라고 적기
        # joint_indices["panda_joint4"]        나중에 꺼내면 → 3
        joint_indices = {}
        link_indices = {}
        # 관절이 몇 개 인지 묻기
        joint_count = p.getNumJoints(
            robot_id, physicsClientId=client_id
        )

        for index in range(joint_count):
            # 관절 하나의 신상카드 받기
            # 관절 하나의에 대한 정보를 17칸짜리 튜플로 반환
            # info[0]은 관절번호
            # info[1] 관절이름
            # info[2] 관절종류
            # info[8] 최소각도
            # info[9] 최대각도
            # info[12] 아래에 매달린 링크 이름
            info = p.getJointInfo(
                robot_id, index,
                physicsClientId=client_id,
            )
            # 뒤에 decode가 붙은 이유는 pybullet이 c++로 만들어져서
            # 디코딩이 필요하기 때문 -> str
            joint_indices[info[1].decode("utf-8")] = index
            link_indices[info[12].decode("utf-8")] = index

        # *는 묶음을 풀어서 넣는 것으로 아래와 같은 형태로 반환
        # needed = ("panda_joint1", ..., "panda_joint7",
        #      "panda_finger_joint1", "panda_finger_joint2") 
        needed = (*ARM_NAMES, *FINGER_NAMES)
        missing = [
            name for name in needed
            if name not in joint_indices
        ]

        if missing or EE_LINK_NAME not in link_indices:
            raise RuntimeError(
                f"Unexpected Panda URDF: missing joints={missing}, "
                f"EE link found={EE_LINK_NAME in link_indices}"
            )

        # 이름 목록을 다시 번호 목록으로 변환
        arm_indices = tuple(
            joint_indices[name] for name in ARM_NAMES
        )
        finger_indices = tuple(
            joint_indices[name] for name in FINGER_NAMES
        )

        # 5. Set initial arm pose (no reaching motion yet)
        # 시작 자세,zip은 각 항목을 맞물려 짝으로 만듬
        # p.resetJointState(robot_id, 0, targetValue=0.0)      # J1 받침 회전 0°
        # p.resetJointState(robot_id, 1, targetValue=-0.785)   # J2 어깨 -45°
        for joint_index, angle in zip(
            arm_indices, HOME_ARM_Q
        ):
            p.resetJointState(
                robot_id,
                joint_index,
                targetValue=angle,
                physicsClientId=client_id,
            )

        # Leave fingers open
        for joint_index in finger_indices:
            p.resetJointState(
                robot_id,
                joint_index,
                targetValue=0.02,
                physicsClientId=client_id,
            )

        return RobotScene(
            client_id=client_id,
            plane_id=plane_id,
            cube_id=cube_id,
            robot_id=robot_id,
            arm_joint_indices=arm_indices,
            finger_joint_indices=finger_indices,
            ee_link_index=link_indices[EE_LINK_NAME],
        )

    except Exception:
        p.disconnect(physicsClientId=client_id)
        raise


def close_robot_scene(scene):
    """Close this scene's PyBullet connection."""

    if p.isConnected(physicsClientId=scene.client_id):
        p.disconnect(physicsClientId=scene.client_id)


def capture_sensor_rgbd(scene):
    """Capture the original overhead RGB-D camera."""

    view = p.computeViewMatrix(
        cameraEyePosition=[0, 0, 1.0],
        cameraTargetPosition=[0, 0, 0],
        cameraUpVector=[0, 1, 0],
    )
    projection = p.computeProjectionMatrixFOV(
        fov=FOV,
        aspect=WIDTH / HEIGHT,
        nearVal=NEAR,
        farVal=FAR,
    )

    image = p.getCameraImage(
        WIDTH,
        HEIGHT,
        viewMatrix=view,
        projectionMatrix=projection,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=scene.client_id,
    )

    # Existing detection expects RGB normalized to [0, 1]
    rgb = (
        np.asarray(image[2], dtype=np.uint8)
        .reshape(HEIGHT, WIDTH, 4)[:, :, :3]
        .astype(np.float32) / 255.0
    )

    depth_buffer = np.asarray(image[3]).reshape(
        HEIGHT, WIDTH
    )
    depth_m = (
        NEAR * FAR /
        (FAR - (FAR - NEAR) * depth_buffer)
    )

    K = compute_intrinsics(WIDTH, HEIGHT, FOV)

    return rgb, depth_m, np.asarray(view, dtype=float), K


def render_overview(scene):
    """Render an inspection view, not a perception input."""

    width, height = 640, 480

    view = p.computeViewMatrix(
        cameraEyePosition=[0.65, -1.25, 1.20],
        cameraTargetPosition=[-0.20, 0.0, 0.25],
        cameraUpVector=[0, 0, 1],
    )
    projection = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=width / height,
        nearVal=0.05,
        farVal=3.0,
    )

    image = p.getCameraImage(
        width,
        height,
        viewMatrix=view,
        projectionMatrix=projection,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=scene.client_id,
    )

    return (
        np.asarray(image[2], dtype=np.uint8)
        .reshape(height, width, 4)[:, :, :3]
    )


def main():
    scene = create_robot_scene()

    try:
        types = {
            p.JOINT_REVOLUTE: "revolute",
            p.JOINT_PRISMATIC: "prismatic",
            p.JOINT_FIXED: "fixed",
        }

        print("Panda joints (arm + fingers + fixed):")

        joint_count = p.getNumJoints(
            scene.robot_id,
            physicsClientId=scene.client_id,
        )

        for index in range(joint_count):
            info = p.getJointInfo(
                scene.robot_id,
                index,
                physicsClientId=scene.client_id,
            )
            joint_type = types.get(
                info[2], str(info[2])
            )
            limits = (
                f"[{info[8]:.3f}, {info[9]:.3f}]"
                if info[2] in (
                    p.JOINT_REVOLUTE, p.JOINT_PRISMATIC
                )
                else "-"
            )

            print(
                f"  {index:2d} {info[1].decode():24s} "
                f"{joint_type:9s} {limits:18s} "
                f"-> {info[12].decode()}"
            )

        # World position of the virtual EE frame
        ee_state = p.getLinkState(
            scene.robot_id,
            scene.ee_link_index,
            computeForwardKinematics=True,
            physicsClientId=scene.client_id,
        )

        rgb, depth_m, view, K = capture_sensor_rgbd(
            scene
        )

        red_pixels = int(np.count_nonzero(
            detect_red_object(rgb)
        ))

        if red_pixels == 0:
            raise RuntimeError(
                "Sensor cannot see the red cube: "
                "inspect robot placement"
            )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        plt.imsave(
            OUTPUT_DIR / "robot_scene_overview.png",
            render_overview(scene),
        )
        plt.imsave(
            OUTPUT_DIR / "sensor_rgb.png", rgb
        )

        np.save(OUTPUT_DIR / "sensor_rgb.npy", rgb)
        np.save(OUTPUT_DIR / "sensor_depth_m.npy", depth_m)
        np.save(OUTPUT_DIR / "sensor_view_matrix.npy", view)
        np.save(OUTPUT_DIR / "sensor_K.npy", K)

        print("\nRobot base (World, m):", ROBOT_BASE)
        print("Arm joints:", len(scene.arm_joint_indices))
        print("Finger joints:", len(scene.finger_joint_indices))
        print(
            "EE link:", EE_LINK_NAME,
            "(index", scene.ee_link_index, ")"
        )
        print(
            "EE initial World position (m):",
            np.round(ee_state[4], 4),
        )
        print("Sensor RGB shape:", rgb.shape)
        print("Sensor depth shape:", depth_m.shape)
        print("Red pixels detected:", red_pixels)
        print("Saved to:", OUTPUT_DIR)

    finally:
        close_robot_scene(scene)


if __name__ == "__main__":
    main()
