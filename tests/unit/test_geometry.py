import math

import numpy as np
import pytest

from motionsense.features.geometry import (
    compute_pose_features,
    head_yaw,
    joint_angle,
    midpoint,
    tilt_from_vertical,
)
from motionsense.simulation.skeleton import LIE, SIT, STAND, build_skeleton
from motionsense.types import KP, Keypoints


def kps_with(points: dict[int, tuple[float, float]], conf: float = 0.9) -> Keypoints:
    xy = np.zeros((17, 2))
    c = np.zeros(17)
    for idx, (x, y) in points.items():
        xy[idx] = (x, y)
        c[idx] = conf
    return Keypoints(xy=xy, conf=c)


class TestPrimitives:
    def test_joint_angle_straight_and_right(self):
        assert joint_angle(np.array([0, 0]), np.array([1, 0]), np.array([2, 0])) == pytest.approx(
            180
        )
        assert joint_angle(np.array([0, 1]), np.array([0, 0]), np.array([1, 0])) == pytest.approx(
            90
        )

    def test_joint_angle_degenerate_is_nan(self):
        assert math.isnan(joint_angle(np.array([0, 0]), np.array([0, 0]), np.array([1, 0])))

    @pytest.mark.parametrize(
        ("top", "bottom", "expected"),
        [
            ((0, 0), (0, 10), 0.0),
            ((10, 0), (0, 0), 90.0),
            ((5, 5), (0, 10), 45.0),
            ((0, 10), (0, 0), 0.0),
        ],
    )
    def test_tilt_is_folded_to_0_90(self, top, bottom, expected):
        assert tilt_from_vertical(np.array(top, float), np.array(bottom, float)) == pytest.approx(
            expected
        )

    def test_midpoint_uses_single_visible_point(self):
        kps = kps_with({KP.L_HIP: (10, 20)})
        assert midpoint(kps, KP.L_HIP, KP.R_HIP, 0.3) == pytest.approx([10, 20])
        assert midpoint(kps, KP.L_KNEE, KP.R_KNEE, 0.3) is None

    def test_head_yaw_from_ears(self):
        centre = kps_with({KP.NOSE: (50, 0), KP.L_EAR: (40, 0), KP.R_EAR: (60, 0)})
        right = kps_with({KP.NOSE: (58, 0), KP.L_EAR: (40, 0), KP.R_EAR: (60, 0)})
        one_ear = kps_with({KP.NOSE: (30, 0), KP.L_EAR: (40, 0)})
        assert head_yaw(centre, 0.3) == pytest.approx(0.0)
        assert head_yaw(right, 0.3) == pytest.approx(0.8)
        assert head_yaw(one_ear, 0.3) == -1.0
        assert math.isnan(head_yaw(kps_with({}), 0.3))


class TestPoseFeatures:
    def test_standing_is_upright_and_straight(self):
        kps, bbox = build_skeleton(STAND)
        f = compute_pose_features(kps, bbox)
        assert f.torso_tilt < 5
        assert f.knee_angle > 160 and f.hip_angle > 160
        assert f.has_torso and f.has_legs
        assert f.torso_len == pytest.approx(0.18 * 720, rel=0.01)

    def test_sitting_has_bent_hips_and_knees(self):
        f = compute_pose_features(*build_skeleton(SIT))
        assert f.hip_angle == pytest.approx(95, abs=3)
        assert f.knee_angle == pytest.approx(95, abs=3)

    def test_lying_is_horizontal(self):
        f = compute_pose_features(*build_skeleton(LIE))
        assert f.torso_tilt > 85

    def test_missing_torso_falls_back_to_bbox_centre(self):
        kps = kps_with({KP.NOSE: (50, 10)})
        f = compute_pose_features(kps, (0, 0, 100, 200))
        assert not f.has_torso and math.isnan(f.torso_tilt)
        assert f.hip_xy == (50.0, 100.0)
        assert f.head_y == 10.0

    def test_keypoints_validate_shape_and_are_readonly(self):
        with pytest.raises(ValueError):
            Keypoints(xy=np.zeros((5, 2)), conf=np.zeros(17))
        with pytest.raises(ValueError):
            Keypoints(xy=np.zeros((17, 2)), conf=np.zeros(5))
        kps = kps_with({})
        with pytest.raises(ValueError):
            kps.xy[0, 0] = 1.0
