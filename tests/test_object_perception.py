
import numpy as np
import pytest

from src.object_perception import (
    detect_red_object,
    select_mask_center_pixel,
)


# Test 1: Detect only red pixels
# 가상의 3x3 RGB이미지를 하나 생성하고 빨간 픽셀 하나만 검출되는 확인
def test_detect_red_object():
    rgb = np.zeros((3, 3, 3), dtype=float)

    rgb[1, 1] = [0.9, 0.1, 0.1]  # Red
    rgb[0, 0] = [0.2, 0.3, 0.9]  # Blue

    mask = detect_red_object(rgb)

    assert mask.shape == (3, 3)
    assert mask.dtype == np.bool_
    assert mask[1, 1]
    assert not mask[0, 0]
    assert np.count_nonzero(mask) == 1


# Test 2: Reject invalid RGB shape
def test_invalid_rgb_shape():
    invalid_rgb = np.zeros((3, 3))

    with pytest.raises(ValueError):
        detect_red_object(invalid_rgb)


# Test 3: Select center pixel
# 3x3 크기의 정사각형 Mask에서 중앙좌표(2,2) 선택하는지 검사 
def test_select_mask_center_pixel():
    mask = np.zeros((5, 5), dtype=bool)

    mask[1:4, 1:4] = True

    u, v = select_mask_center_pixel(mask)

    assert (u, v) == (2, 2)
    assert mask[v, u]


# Test 4: Selected pixel must belong to mask
# 검출된 픽셀이 떨어져 있어 평균 위치가 검출 영역 밖이더라도
# 실제 Mask 안의 픽셀을 반환하는지 확인
def test_selected_pixel_inside_mask():
    mask = np.zeros((5, 5), dtype=bool)

    mask[1, 1] = True
    mask[1, 3] = True

    # Centroid is (u=2, v=1), outside the mask.
    u, v = select_mask_center_pixel(mask)

    assert mask[v, u]
    assert (u, v) in [(1, 1), (3, 1)]


# Test 5: Empty mask must raise an error
def test_empty_mask():
    mask = np.zeros((5, 5), dtype=bool)

    with pytest.raises(ValueError):
        select_mask_center_pixel(mask)
