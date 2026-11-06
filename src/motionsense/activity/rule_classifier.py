"""Rule-based fuzzy activity classifier (v1).

Each activity gets a membership score in [0, 1] built from smooth sigmoid "soft thresholds"
over scale-invariant features. Scores are normalised into probabilities. No training data is
needed, and every number below is an interpretable threshold you can tune.
"""

from __future__ import annotations

import math

from motionsense.activity.base import ACTIVITIES, ActivityContext, ActivityResult
from motionsense.features.geometry import PoseFeatures
from motionsense.features.kinematics import MotionFeatures

SEAT_SUPPORTS = frozenset({"chair", "couch", "bench", "toilet", "bed"})
MIN_TOTAL_SCORE = 1e-3


def high(x: float, centre: float, width: float) -> float:
    """Soft 'x is greater than centre'. NaN -> 0.5 (unknown)."""
    if math.isnan(x):
        return 0.5
    z = (x - centre) / width
    if z > 40:
        return 1.0
    if z < -40:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def low(x: float, centre: float, width: float) -> float:
    """Soft 'x is less than centre'. NaN -> 0.5 (unknown)."""
    return 1.0 - high(x, centre, width) if not math.isnan(x) else 0.5


def _tilt(pose: PoseFeatures) -> float:
    """Best available body tilt: torso, else head-over-shoulders, else whole-body box shape.

    NaN when none of these is trustworthy (e.g. a close-up face), so the classifier stays
    uncertain instead of mistaking a wide close-up box for someone lying down.
    """
    if not math.isnan(pose.torso_tilt):
        return pose.torso_tilt
    return pose.posture_tilt


class RuleBasedClassifier:
    """Interpretable fuzzy-logic classifier over pose and motion features."""

    def __init__(self, looking_around_yaw_std: float = 0.28) -> None:
        self._yaw_std_threshold = looking_around_yaw_std

    def classify(
        self, pose: PoseFeatures, motion: MotionFeatures, context: ActivityContext
    ) -> ActivityResult:
        tilt = _tilt(pose)
        upright = low(tilt, 35.0, 6.0)
        semi_upright = low(tilt, 50.0, 7.0)
        horizontal = high(tilt, 55.0, 6.0)

        # Locomotion = moving across the view OR legs alternating (walking towards the camera).
        runish = max(high(motion.horizontal_speed, 3.0, 0.4), high(motion.gait, 7.0, 1.0))
        walkish = max(high(motion.horizontal_speed, 0.6, 0.12), high(motion.gait, 1.2, 0.3))
        walkish *= 1.0 - runish
        still = low(max(motion.horizontal_speed, 0.5 * motion.gait), 0.55, 0.12)

        on_seat = context.support in SEAT_SUPPORTS

        if pose.has_legs:
            scores, quality = self._full_body(
                pose, motion, upright, semi_upright, horizontal, still, walkish, runish, on_seat
            )
        else:
            tilt_known = not math.isnan(tilt)
            scores, quality = self._upper_body(
                upright
                if tilt_known
                else 0.8,  # unknown tilt, but a visible head/torso: upright-ish
                horizontal if tilt_known else 0.0,
                still,
                walkish,
                runish,
                on_seat,
            )

        if pose.quality < 0.25:
            quality = "low"

        total = sum(scores.values())
        if total < MIN_TOTAL_SCORE:
            probs = {a: 1.0 / len(ACTIVITIES) for a in ACTIVITIES}
            quality = "low"
        else:
            probs = {a: scores[a] / total for a in ACTIVITIES}

        return ActivityResult(
            probabilities=probs, quality=quality, attributes=self._attributes(pose, motion)
        )

    @staticmethod
    def _full_body(
        pose: PoseFeatures,
        motion: MotionFeatures,
        upright: float,
        semi_upright: float,
        horizontal: float,
        still: float,
        walkish: float,
        runish: float,
        on_seat: bool,
    ) -> tuple[dict[str, float], str]:
        knee_straight = high(pose.knee_angle, 145.0, 8.0)
        hip_straight = high(pose.hip_angle, 145.0, 8.0)
        hip_bent = low(pose.hip_angle, 135.0, 10.0)
        knee_bent = low(pose.knee_angle, 140.0, 10.0)
        deep_knee = low(pose.knee_angle, 105.0, 8.0)
        squat_motion = high(motion.knee_range, 30.0, 8.0)
        pushup_motion = high(motion.elbow_range, 25.0, 7.0)
        arms_support = high(pose.wrist_support, 0.5, 0.15)
        plank = hip_straight * knee_straight * arms_support

        seat_bonus = 1.6 if on_seat else 1.0
        # Repeated knee bending without leg alternation = a squat set (incl. the top of each rep).
        squat_set = (
            semi_upright * squat_motion * low(motion.gait, 1.2, 0.3) * (0.0 if on_seat else 0.9)
        )
        scores = {
            "standing": upright * knee_straight * hip_straight * still * (1.0 - 0.8 * squat_set),
            "sitting": semi_upright
            * hip_bent
            * knee_bent
            * still
            * (1.0 - 0.7 * squat_motion)
            * seat_bonus,
            "lying": horizontal * (1.0 - pushup_motion) * (1.0 - 0.7 * plank),
            "walking": upright * walkish * (0.4 + 0.6 * knee_straight),
            "running": upright * runish,
            "squatting": max(
                semi_upright
                * deep_knee
                * hip_bent
                * (0.35 + 0.65 * squat_motion)
                * (0.4 if on_seat else 1.0),
                squat_set,
            ),
            "push_up": horizontal * max(pushup_motion, 0.6 * plank),
        }
        return scores, "full"

    @staticmethod
    def _upper_body(
        upright: float,
        horizontal: float,
        still: float,
        walkish: float,
        runish: float,
        on_seat: bool,
    ) -> tuple[dict[str, float], str]:
        """Only head/torso visible (typical laptop webcam at a desk): be modest."""
        sit_prior = 0.75 if on_seat else 0.55
        scores = {
            "standing": upright * still * (1.0 - sit_prior),
            "sitting": upright * still * sit_prior,
            "lying": horizontal,
            "walking": upright * walkish,
            "running": upright * runish,
            "squatting": 0.0,
            "push_up": 0.0,
        }
        return scores, "partial"

    def _attributes(self, pose: PoseFeatures, motion: MotionFeatures) -> dict[str, object]:
        yaw = pose.head_yaw
        if math.isnan(yaw):
            direction = "unknown"
        elif yaw < -0.35:
            direction = "left"
        elif yaw > 0.35:
            direction = "right"
        else:
            direction = "forward"
        return {
            "looking_around": bool(motion.yaw_std >= self._yaw_std_threshold),
            "head_direction": direction,
        }
