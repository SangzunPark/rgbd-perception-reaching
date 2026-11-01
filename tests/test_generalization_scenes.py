
import numpy as np
import pytest

from src.generate_generalization_scenes import (
    POSITION_GROUPS,
    CAMERA_DIRECTIONS,
    camera_eye,
    inspect_scene,
    render_scene,
    validate_position_groups,
)


def test_position_groups():
    validate_position_groups()

    assert len(POSITION_GROUPS["development"]) == 9
    assert len(POSITION_GROUPS["interpolation"]) == 16
    assert len(POSITION_GROUPS["extrapolation"]) == 8


def test_camera_viewpoints():
    assert len(CAMERA_DIRECTIONS) == 4

    for camera_name in CAMERA_DIRECTIONS:
        eye = camera_eye(camera_name)

        assert np.linalg.norm(eye) == pytest.approx(1.0)


def test_rendered_scene_integrity(tmp_path):
    scene_dir = tmp_path / "scene_01"
    position = (0.025, 0.0125)

    created = render_scene(
        position=position,
        camera_name="oblique_x",
        output_dir=scene_dir,
    )

    assert created is True

    result = inspect_scene(
        scene_dir,
        position,
        "oblique_x",
    )

    assert result["rgb_detected_pixels"] > 0
    assert result["segmentation_pixels"] > 0
    assert len(result["gt_top_world_m"]) == 3
    assert len(result["gt_pixel_uv"]) == 2

    # Re-running must not overwrite a completed scene.
    second = render_scene(
        position=position,
        camera_name="oblique_x",
        output_dir=scene_dir,
    )

    assert second is False


def test_unknown_camera_rejected():
    with pytest.raises(ValueError):
        camera_eye("unknown_camera")
