"""Per-person fall detection state machine.

    MONITORING --impact--> IMPACT --stays down confirm_s--> FALLEN --down inactive_alert_s--> INACTIVE
         ^                    |                                |                                |
         +------ recovered ---+------------ recovered ---------+----------- recovered ----------+

*Impact* = within ~1 s the hips drop fast and far AND the torso rotates towards horizontal,
starting from an upright posture. Speed is what separates a fall from lying down on purpose.
Landing on a bed/couch strongly reduces confidence. Lying still on the floor for a long time
raises an alert even if the fall itself was not seen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from motionsense.activity.rule_classifier import high, low
from motionsense.config import FallConfig
from motionsense.events.models import EventType
from motionsense.fall.context import LYING_SUPPORTS
from motionsense.features.geometry import PoseFeatures
from motionsense.features.kinematics import MotionFeatures

SEAT_SUPPORTS = frozenset({"chair", "bench", "toilet", "couch"})
STILL_ENERGY = 0.6  # torso lengths/s of mean keypoint motion
REARM_AFTER_RECOVERY_S = 2.0
UNSEEN_FALL_LYING_PROB = 0.6
RECOVERY_HIP_RISE = 0.7  # torso lengths
LOST_MIN_TILT_FRACTION = 0.5


class FallState(StrEnum):
    MONITORING = "monitoring"
    IMPACT = "impact"
    FALLEN = "fallen"
    INACTIVE = "inactive"


@dataclass(frozen=True, slots=True)
class FallSignal:
    type: EventType
    confidence: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FallUpdate:
    state: FallState
    risk: float  # 0..1, for the live gauge
    impact_score: float
    signals: tuple[FallSignal, ...] = ()


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


class FallDetector:
    def __init__(self, config: FallConfig) -> None:
        self.cfg = config
        self.state = FallState.MONITORING
        self._t_impact = 0.0
        self._impact_peak = 0.0
        self._recovered_since: float | None = None
        self._rearm_at = -math.inf
        self._last_suspected = -math.inf
        self._floor_still_since: float | None = None
        self._lowest_hip_y = -math.inf  # image y grows downwards: larger = lower
        self._hip_y = 0.0
        self._last_motion: MotionFeatures | None = None
        self._risk = 0.0

    # ------------------------------------------------------------------ scoring
    def impact_score(self, motion: MotionFeatures, support: str | None) -> float:
        cfg = self.cfg
        s_drop = high(motion.drop_1s, cfg.drop_threshold, 0.15 * cfg.drop_threshold)
        s_tilt = high(motion.tilt_change, cfg.tilt_change_threshold, 6.0)
        s_vel = high(
            motion.peak_down_velocity, cfg.velocity_threshold, 0.25 * cfg.velocity_threshold
        )
        was_upright = low(motion.min_recent_tilt, 40.0, 6.0)
        # Fuzzy AND: a fall needs a big drop AND a big rotation AND high speed together.
        score = s_drop * s_tilt * s_vel * was_upright
        if support in LYING_SUPPORTS:
            score *= cfg.furniture_factor
        return _clip(score)

    @staticmethod
    def _is_recovered(probs: dict[str, float], support: str | None) -> bool:
        upright = probs.get("standing", 0.0) + probs.get("walking", 0.0) + probs.get("running", 0.0)
        if upright >= 0.5:
            return True
        return support in SEAT_SUPPORTS and probs.get("sitting", 0.0) >= 0.5

    def _got_up(self, pose: PoseFeatures, motion: MotionFeatures, probs: dict[str, float]) -> bool:
        """Hips rose well above their lowest point after the fall and the torso is upright:
        e.g. the person climbed into a chair the object detector did not see."""
        if self.state == FallState.MONITORING or math.isnan(pose.torso_tilt):
            return False
        rise = (self._lowest_hip_y - pose.hip_xy[1]) / max(motion.scale, 1.0)
        upright_posture = probs.get("standing", 0.0) + probs.get("sitting", 0.0) >= 0.5
        return rise >= RECOVERY_HIP_RISE and pose.torso_tilt < 40.0 and upright_posture

    def _confirm_confidence(self, probs: dict[str, float]) -> float:
        bonus = 0.05 if probs.get("lying", 0.0) > 0.5 else 0.0
        return _clip(0.55 + 0.4 * self._impact_peak + bonus, 0.0, 0.99)

    # ------------------------------------------------------------------ update
    def update(
        self,
        ts: float,
        pose: PoseFeatures,
        motion: MotionFeatures,
        probs: dict[str, float],
        support: str | None,
    ) -> FallUpdate:
        self._last_motion = motion
        impact = self.impact_score(motion, support)
        self._hip_y = pose.hip_xy[1]
        if self.state != FallState.MONITORING:
            self._lowest_hip_y = max(self._lowest_hip_y, pose.hip_xy[1])
        recovered = self._is_recovered(probs, support) or self._got_up(pose, motion, probs)
        if recovered:
            self._recovered_since = (
                self._recovered_since if self._recovered_since is not None else ts
            )
        else:
            self._recovered_since = None
        held = (
            self._recovered_since is not None
            and ts - self._recovered_since >= self.cfg.recover_hold_s
        )

        handler = {
            FallState.MONITORING: self._monitoring,
            FallState.IMPACT: self._impact,
            FallState.FALLEN: self._down,
            FallState.INACTIVE: self._down,
        }[self.state]
        signals, target_risk = handler(ts, impact, probs, support, motion, held)

        # Smooth the gauge slightly so it is readable, but let alarms jump up instantly.
        self._risk = (
            target_risk if target_risk > self._risk else 0.7 * self._risk + 0.3 * target_risk
        )
        return FallUpdate(
            state=self.state, risk=_clip(self._risk), impact_score=impact, signals=tuple(signals)
        )

    def _monitoring(
        self,
        ts: float,
        impact: float,
        probs: dict[str, float],
        support: str | None,
        motion: MotionFeatures,
        held: bool,
    ) -> tuple[list[FallSignal], float]:
        signals: list[FallSignal] = []
        if impact >= self.cfg.impact_score_threshold and ts >= self._rearm_at:
            self.state = FallState.IMPACT
            self._t_impact = ts
            self._impact_peak = impact
            self._lowest_hip_y = self._hip_y
            self._recovered_since = None
            if self.cfg.emit_suspected and ts - self._last_suspected >= self.cfg.cooldown_s:
                self._last_suspected = ts
                signals.append(
                    FallSignal(
                        EventType.FALL_SUSPECTED,
                        impact,
                        {"reason": "rapid_descent", **_motion_details(motion)},
                    )
                )
            return signals, max(impact, 0.5)

        # A person found lying still on the floor (fall not witnessed) is also an incident.
        lying_on_floor = (
            probs.get("lying", 0.0) >= UNSEEN_FALL_LYING_PROB and support not in LYING_SUPPORTS
        )
        if lying_on_floor and motion.motion_energy < STILL_ENERGY:
            self._floor_still_since = (
                self._floor_still_since if self._floor_still_since is not None else ts
            )
            still_for = ts - self._floor_still_since
            if still_for >= self.cfg.inactive_alert_s:
                self.state = FallState.INACTIVE
                self._t_impact = self._floor_still_since
                self._impact_peak = 0.0
                signals.append(
                    FallSignal(
                        EventType.INACTIVE_ON_FLOOR,
                        0.7,
                        {"reason": "lying_still_on_floor", "duration_s": round(still_for, 1)},
                    )
                )
                return signals, 0.9
            return signals, max(0.8 * impact, 0.15 + 0.35 * still_for / self.cfg.inactive_alert_s)
        self._floor_still_since = None
        return signals, 0.8 * impact

    def _impact(
        self,
        ts: float,
        impact: float,
        probs: dict[str, float],
        support: str | None,
        motion: MotionFeatures,
        held: bool,
    ) -> tuple[list[FallSignal], float]:
        elapsed = ts - self._t_impact
        if elapsed <= 1.0:
            self._impact_peak = max(self._impact_peak, impact)
        if held:
            self._reset(ts)
            return [
                FallSignal(
                    EventType.FALL_RECOVERED,
                    0.9,
                    {"stage": "before_confirmation", "down_s": round(elapsed, 1)},
                )
            ], 0.2
        if elapsed < self.cfg.confirm_s:
            return [], 0.5 + 0.35 * (elapsed / self.cfg.confirm_s) * max(self._impact_peak, 0.5)

        confidence = self._confirm_confidence(probs)
        if support in LYING_SUPPORTS:
            confidence *= self.cfg.furniture_factor
            if confidence < self.cfg.impact_score_threshold:
                self._reset(ts)
                return [
                    FallSignal(
                        EventType.FALL_RECOVERED,
                        0.8,
                        {"stage": "dismissed", "reason": f"landed_on_{support}"},
                    )
                ], 0.1
        self.state = FallState.FALLEN
        return [
            FallSignal(
                EventType.FALL_CONFIRMED,
                confidence,
                {
                    "reason": "stayed_down_after_impact",
                    "down_s": round(elapsed, 1),
                    "impact_peak": round(self._impact_peak, 3),
                    "support": support,
                },
            )
        ], confidence

    def _down(
        self,
        ts: float,
        impact: float,
        probs: dict[str, float],
        support: str | None,
        motion: MotionFeatures,
        held: bool,
    ) -> tuple[list[FallSignal], float]:
        down_s = ts - self._t_impact
        if held:
            stage = "after_confirmation" if self.state == FallState.FALLEN else "after_inactivity"
            self._reset(ts)
            return [
                FallSignal(
                    EventType.FALL_RECOVERED, 0.9, {"stage": stage, "down_s": round(down_s, 1)}
                )
            ], 0.2
        if self.state == FallState.FALLEN and down_s >= self.cfg.inactive_alert_s:
            self.state = FallState.INACTIVE
            return [
                FallSignal(
                    EventType.INACTIVE_ON_FLOOR,
                    0.9,
                    {"reason": "not_up_after_fall", "duration_s": round(down_s, 1)},
                )
            ], 1.0
        return [], 1.0 if self.state == FallState.INACTIVE else 0.9

    def on_track_lost(self, ts: float) -> FallUpdate:
        """The person vanished (left the frame or became occluded)."""
        signals: list[FallSignal] = []
        motion = self._last_motion
        if self.state == FallState.IMPACT:
            conf = 0.8 * (0.55 + 0.4 * self._impact_peak)
            signals.append(
                FallSignal(EventType.FALL_CONFIRMED, conf, {"reason": "track_lost_after_impact"})
            )
            self.state = FallState.FALLEN
        elif (
            self.state == FallState.MONITORING
            and motion is not None
            and motion.drop_1s >= self.cfg.drop_threshold
            and motion.peak_down_velocity >= self.cfg.velocity_threshold
            # the body must also have started to tip over: walking or leaning out of view
            # (e.g. stepping close past the camera) moves the box without rotating the torso
            and motion.tilt_change >= LOST_MIN_TILT_FRACTION * self.cfg.tilt_change_threshold
            and ts - self._last_suspected >= self.cfg.cooldown_s
        ):
            self._last_suspected = ts
            signals.append(
                FallSignal(
                    EventType.FALL_SUSPECTED,
                    0.5,
                    {"reason": "lost_during_descent", **_motion_details(motion)},
                )
            )
        return FallUpdate(
            state=self.state, risk=self._risk, impact_score=0.0, signals=tuple(signals)
        )

    def _reset(self, ts: float) -> None:
        self.state = FallState.MONITORING
        self._lowest_hip_y = -math.inf
        self._impact_peak = 0.0
        self._recovered_since = None
        self._floor_still_since = None
        self._rearm_at = ts + REARM_AFTER_RECOVERY_S


def _motion_details(m: MotionFeatures) -> dict[str, float]:
    return {
        "drop": round(m.drop_1s, 3),
        "tilt_change": round(m.tilt_change, 1),
        "peak_down_velocity": round(m.peak_down_velocity, 3),
    }
