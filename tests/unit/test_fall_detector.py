"""State-machine tests for FallDetector, driven by hand-crafted feature streams."""

from dataclasses import replace

import pytest

from motionsense.config import FallConfig
from motionsense.events.models import EventType
from motionsense.fall.detector import FallDetector, FallState
from motionsense.features.geometry import compute_pose_features
from motionsense.features.kinematics import MotionFeatures
from motionsense.simulation.skeleton import STAND, build_skeleton

POSE = compute_pose_features(*build_skeleton(STAND))
CALM = MotionFeatures(
    scale=130.0,
    body_speed=0.0,
    horizontal_speed=0.0,
    vertical_velocity=0.0,
    drop_1s=0.0,
    peak_down_velocity=0.0,
    tilt_change=0.0,
    min_recent_tilt=2.0,
    gait=0.0,
    motion_energy=0.1,
    elbow_range=0.0,
    knee_range=0.0,
    yaw_std=0.0,
    span=3.0,
)
IMPACT = replace(CALM, drop_1s=1.4, tilt_change=75.0, peak_down_velocity=3.5, min_recent_tilt=3.0)
STANDING = {"standing": 0.9, "lying": 0.05, "sitting": 0.05}
LYING = {"standing": 0.02, "lying": 0.9, "sitting": 0.08}


def make(**cfg) -> FallDetector:
    return FallDetector(FallConfig(**cfg))


def types(update) -> list[EventType]:
    return [s.type for s in update.signals]


def fall_and_stay_down(det: FallDetector, until: float, support=None, dt: float = 0.1):
    updates = [det.update(0.0, POSE, IMPACT, LYING, support)]
    t = dt
    while t <= until + 1e-9:
        updates.append(det.update(t, POSE, CALM, LYING, support))
        t += dt
    return updates


def all_types(updates) -> list[EventType]:
    return [s.type for u in updates for s in u.signals]


def test_calm_standing_never_alerts():
    det = make()
    for i in range(100):
        u = det.update(i * 0.1, POSE, CALM, STANDING, None)
        assert u.signals == ()
        assert u.risk < 0.1
    assert det.state == FallState.MONITORING


def test_impact_then_staying_down_confirms_fall():
    det = make(confirm_s=2.0)
    updates = fall_and_stay_down(det, until=2.5)
    assert types(updates[0]) == [EventType.FALL_SUSPECTED]
    assert updates[0].state == FallState.IMPACT
    assert all_types(updates).count(EventType.FALL_CONFIRMED) == 1
    confirmed = next(s for u in updates for s in u.signals if s.type == EventType.FALL_CONFIRMED)
    assert confirmed.confidence > 0.8
    assert det.state == FallState.FALLEN
    assert updates[-1].risk >= 0.85


def test_suspected_can_be_disabled():
    det = make(emit_suspected=False)
    assert det.update(0.0, POSE, IMPACT, LYING, None).signals == ()
    assert det.state == FallState.IMPACT


def test_getting_up_quickly_is_recovery_not_fall():
    det = make(confirm_s=2.0, recover_hold_s=0.5)
    det.update(0.0, POSE, IMPACT, LYING, None)
    out = [det.update(t / 10, POSE, CALM, STANDING, None) for t in range(1, 10)]
    assert EventType.FALL_RECOVERED in all_types(out)
    assert EventType.FALL_CONFIRMED not in all_types(out)
    assert det.state == FallState.MONITORING


def test_rearm_delay_after_recovery():
    det = make(recover_hold_s=0.0)
    det.update(0.0, POSE, IMPACT, LYING, None)
    det.update(0.1, POSE, CALM, STANDING, None)  # recovered immediately
    assert det.state == FallState.MONITORING
    assert det.update(0.2, POSE, IMPACT, LYING, None).signals == ()  # still within re-arm window
    assert types(det.update(3.0, POSE, IMPACT, LYING, None)) == [EventType.FALL_SUSPECTED]


def test_landing_on_bed_is_dismissed():
    det = make(furniture_factor=0.35)
    # The impact is already damped by the bed, so no event at all.
    updates = fall_and_stay_down(det, until=3.0, support="bed")
    assert EventType.FALL_CONFIRMED not in all_types(updates)


def test_landing_next_to_bed_then_on_it_is_dismissed_at_confirmation():
    det = make(confirm_s=1.0)
    det.update(0.0, POSE, IMPACT, LYING, None)  # impact seen off-furniture
    out = [det.update(t / 10, POSE, CALM, LYING, "couch") for t in range(1, 15)]
    recovered = [s for u in out for s in u.signals if s.type == EventType.FALL_RECOVERED]
    assert recovered and recovered[0].details["stage"] == "dismissed"
    assert det.state == FallState.MONITORING


def test_long_time_down_escalates_once_then_recovers():
    det = make(confirm_s=1.0, inactive_alert_s=5.0, recover_hold_s=0.0)
    updates = fall_and_stay_down(det, until=8.0)
    assert all_types(updates).count(EventType.INACTIVE_ON_FLOOR) == 1
    assert det.state == FallState.INACTIVE
    u = det.update(9.0, POSE, CALM, STANDING, None)
    assert types(u) == [EventType.FALL_RECOVERED]
    assert u.signals[0].details["stage"] == "after_inactivity"


def test_lying_still_on_floor_without_seeing_fall_alerts():
    det = make(inactive_alert_s=3.0)
    out = [det.update(t / 10, POSE, CALM, LYING, None) for t in range(0, 40)]
    assert all_types(out) == [EventType.INACTIVE_ON_FLOOR]
    risks = [u.risk for u in out]
    assert risks[20] > risks[2]  # gauge rises while lying still


def test_lying_on_bed_for_long_is_fine():
    det = make(inactive_alert_s=3.0)
    out = [det.update(t / 10, POSE, CALM, LYING, "bed") for t in range(0, 60)]
    assert all_types(out) == []


def test_track_lost_after_impact_confirms():
    det = make()
    det.update(0.0, POSE, IMPACT, LYING, None)
    u = det.on_track_lost(0.5)
    assert types(u) == [EventType.FALL_CONFIRMED]
    assert u.signals[0].details["reason"] == "track_lost_after_impact"


def test_track_lost_during_fast_descent_is_suspected():
    det = make()
    # fast drop and the torso has started to tip (25 deg) when the person disappears
    tipping = replace(IMPACT, tilt_change=25.0)
    det.update(0.0, POSE, tipping, LYING, None)
    assert det.state == FallState.MONITORING
    u = det.on_track_lost(0.5)
    assert types(u) == [EventType.FALL_SUSPECTED]
    assert u.signals[0].details["reason"] == "lost_during_descent"


def test_leaving_the_view_without_tipping_over_is_silent():
    """E.g. stepping close past the camera: the box jumps but the body stays upright."""
    det = make()
    det.update(0.0, POSE, replace(IMPACT, tilt_change=3.0), STANDING, None)
    assert det.on_track_lost(0.5).signals == ()


def test_track_lost_while_calm_is_silent():
    det = make()
    det.update(0.0, POSE, CALM, STANDING, None)
    assert det.on_track_lost(1.0).signals == ()


@pytest.mark.parametrize("support,factor", [(None, 1.0), ("bed", 0.35), ("chair", 1.0)])
def test_impact_score_furniture_factor(support, factor):
    det = make(furniture_factor=0.35)
    assert det.impact_score(IMPACT, support) == pytest.approx(
        det.impact_score(IMPACT, None) * factor
    )
