import math
from dataclasses import replace

import pytest

from motionsense.activity.base import ACTIVITIES, ActivityContext
from motionsense.activity.rep_counter import PUSH_UP, RepCounter
from motionsense.activity.rule_classifier import RuleBasedClassifier, high, low
from motionsense.activity.smoothing import ProbabilitySmoother, StableLabel
from motionsense.features.geometry import compute_pose_features
from motionsense.features.kinematics import MotionFeatures
from motionsense.simulation.skeleton import LIE, PLANK, SIT, STAND, build_skeleton

STILL = MotionFeatures(
    scale=130.0,
    body_speed=0.0,
    horizontal_speed=0.0,
    vertical_velocity=0.0,
    drop_1s=0.0,
    peak_down_velocity=0.0,
    tilt_change=0.0,
    min_recent_tilt=0.0,
    gait=0.0,
    motion_energy=0.0,
    elbow_range=0.0,
    knee_range=0.0,
    yaw_std=0.0,
    span=3.0,
)
CLF = RuleBasedClassifier()


def classify(pose_params, motion=STILL, support=None):
    pose = compute_pose_features(*build_skeleton(pose_params))
    return CLF.classify(pose, motion, ActivityContext(support=support))


def test_soft_thresholds():
    assert high(10, 5, 1) > 0.99
    assert high(0, 5, 1) < 0.01
    assert high(5, 5, 1) == pytest.approx(0.5)
    assert low(0, 5, 1) > 0.99
    assert high(float("nan"), 5, 1) == 0.5 and low(float("nan"), 5, 1) == 0.5
    assert high(1e9, 0, 1) == 1.0 and high(-1e9, 0, 1) == 0.0


@pytest.mark.parametrize(
    ("pose", "motion_changes", "expected"),
    [
        (STAND, {}, "standing"),
        (SIT, {}, "sitting"),
        (LIE, {}, "lying"),
        (STAND, {"horizontal_speed": 1.4, "gait": 3.0}, "walking"),
        (STAND, {"horizontal_speed": 4.0, "gait": 12.0}, "running"),
        (STAND, {"gait": 2.5}, "walking"),  # walking towards the camera
        (PLANK, {"elbow_range": 60.0}, "push_up"),
    ],
)
def test_classifies_basic_activities(pose, motion_changes, expected):
    result = classify(pose, replace(STILL, **motion_changes))
    top, prob = result.top
    assert top == expected
    assert prob > 0.5
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert set(result.probabilities) == set(ACTIVITIES)
    assert result.quality == "full"


def test_seat_context_boosts_sitting():
    without = classify(SIT).probabilities["sitting"]
    with_chair = classify(SIT, support="chair").probabilities["sitting"]
    assert with_chair >= without


def test_upper_body_only_is_partial_and_modest():
    pose = compute_pose_features(*build_skeleton(STAND))
    upper = replace(pose, has_legs=False, hip_angle=math.nan, knee_angle=math.nan)
    result = CLF.classify(upper, STILL, ActivityContext())
    assert result.quality == "partial"
    assert result.top[0] == "sitting"
    assert result.top[1] < 0.8


def test_low_evidence_gives_uniform_low_quality():
    pose = compute_pose_features(*build_skeleton(STAND))
    weird = replace(pose, torso_tilt=45.0, hip_angle=140.0, knee_angle=150.0, quality=0.1)
    result = CLF.classify(weird, replace(STILL, horizontal_speed=0.6), ActivityContext())
    assert result.quality == "low"


def test_head_direction_and_looking_around_attributes():
    result = classify(STAND.with_(yaw=0.9), replace(STILL, yaw_std=0.5))
    assert result.attributes == {"looking_around": True, "head_direction": "right"}
    assert classify(STAND.with_(yaw=-0.9)).attributes["head_direction"] == "left"
    assert classify(STAND).attributes["head_direction"] == "forward"


class TestSmoothing:
    def test_ema_converges_and_normalises(self):
        s = ProbabilitySmoother(tau_s=0.5)
        s.update({"a": 1.0, "b": 0.0}, 0.0)
        out = {}
        for i in range(1, 40):
            out = s.update({"a": 0.0, "b": 1.0}, i * 0.1)
        assert out["b"] > 0.99
        assert sum(out.values()) == pytest.approx(1.0)

    def test_ema_rejects_bad_tau(self):
        with pytest.raises(ValueError):
            ProbabilitySmoother(0)

    def test_stable_label_requires_hold(self):
        lbl = StableLabel(hold_s=1.0)
        assert lbl.update("sit", 0.0) is None
        assert lbl.update("sit", 0.5) is None
        assert lbl.update("sit", 1.0) == "sit"
        assert lbl.update("sit", 1.5) is None
        assert lbl.update("stand", 2.0) is None
        assert lbl.update("sit", 2.2) is None  # flicker back resets the candidate
        assert lbl.current == "sit"


class TestRepCounter:
    def test_counts_full_cycles_only_when_activity_likely(self):
        c = RepCounter(PUSH_UP)
        angles = [170, 90, 170, 90, 170, 120, 170]
        reps = [c.update(a, 0.9) for a in angles]
        assert c.count == 2
        assert reps.count(True) == 2

    def test_ignores_nan_and_unlikely_activity(self):
        c = RepCounter(PUSH_UP)
        assert c.update(float("nan"), 0.9) is False
        for a in [170, 80, 170]:
            c.update(a, 0.0)
        assert c.count == 0
