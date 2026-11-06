import pytest

from motionsense.features.geometry import compute_pose_features
from motionsense.features.kinematics import compute_motion_features
from motionsense.simulation.scenarios import SCENARIOS, Scenario
from motionsense.simulation.skeleton import build_skeleton
from motionsense.tracking.history import Sample, TrackHistory

FPS = 15.0


def history_until(scenario: Scenario, t_end: float, t_start: float = 0.0) -> TrackHistory:
    hist = TrackHistory()
    t = t_start
    while t <= t_end + 1e-9:
        kps, bbox = build_skeleton(scenario.pose_at(t))
        hist.add(Sample(ts=t, pose=compute_pose_features(kps, bbox), keypoints=kps, bbox=bbox))
        t += 1.0 / FPS
    return hist


def test_empty_history_raises():
    with pytest.raises(ValueError):
        compute_motion_features(TrackHistory())


def test_standing_still_has_no_motion():
    m = compute_motion_features(history_until(SCENARIOS["standing"], 3.0))
    assert m.body_speed == pytest.approx(0.0, abs=1e-6)
    assert m.gait == pytest.approx(0.0, abs=1e-6)
    assert m.drop_1s == pytest.approx(0.0, abs=1e-6)
    assert m.scale == pytest.approx(0.18 * 720, rel=0.01)


def test_fall_produces_large_drop_rotation_and_speed():
    m = compute_motion_features(history_until(SCENARIOS["fall"], 3.6))
    assert m.drop_1s > 0.8
    assert m.tilt_change > 40
    assert m.peak_down_velocity > 1.6
    assert m.min_recent_tilt < 15


def test_walking_has_gait_but_squats_do_not():
    walk = compute_motion_features(history_until(SCENARIOS["walk"], 3.0))
    squat = compute_motion_features(history_until(SCENARIOS["squats"], 4.0))
    assert walk.gait > 1.5
    assert walk.horizontal_speed > 0.8
    assert squat.gait < 0.3
    assert squat.knee_range > 30


def test_history_trims_and_replaces_out_of_order_samples():
    hist = history_until(SCENARIOS["standing"], 7.9)
    assert hist.latest is not None
    assert hist.latest.ts - next(iter(hist)).ts <= 6.0 + 1e-9
    n = len(hist)
    hist.add(hist.latest)  # duplicate timestamp replaces, does not grow
    assert len(hist) == n
    assert hist.window(0.5)[-1] is hist.latest
    assert TrackHistory().window(1.0) == []
    assert TrackHistory().latest is None
