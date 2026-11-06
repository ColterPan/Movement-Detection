"""Temporal (motion) features computed from a track's recent history.

Speeds are expressed in *torso lengths per second* so they do not depend on the
distance between the person and the camera.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from motionsense.tracking.history import Sample, TrackHistory
from motionsense.types import KP

SCALE_WINDOW_S = 5.0
SPEED_WINDOW_S = 0.5
GAIT_WINDOW_S = 0.8
VERTICAL_WINDOW_S = 0.3
IMPACT_WINDOW_S = 1.0
TILT_WINDOW_S = 1.5
OSCILLATION_WINDOW_S = 2.0
MIN_PAIR_DT_S = 0.12
MIN_POINT_CONF = 0.3
LEG_PAIRS = ((KP.L_KNEE, KP.R_KNEE), (KP.L_ANKLE, KP.R_ANKLE))


@dataclass(frozen=True, slots=True)
class MotionFeatures:
    scale: float  # px per torso length (reference)
    body_speed: float  # hip-centre speed (net displacement over 0.5 s)
    horizontal_speed: float  # hip-centre horizontal speed (locomotion across the view)
    vertical_velocity: float  # + = moving down the image
    drop_1s: float  # how far the hips went down in the last second
    peak_down_velocity: float  # fastest downward hip speed in the last second
    tilt_change: float  # current tilt minus the minimum tilt in the last 1.5 s (deg)
    min_recent_tilt: float  # deg
    gait: float  # left/right leg alternation speed (walking/running signature)
    motion_energy: float  # median keypoint speed (all visible points)
    elbow_range: float  # deg, last 2 s
    knee_range: float  # deg, last 2 s
    yaw_std: float  # head-yaw standard deviation, last 2 s
    span: float  # seconds of history available

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def reference_scale(samples: Sequence[Sample]) -> float:
    """Robust torso length: a high percentile over the scale window (foreshortening-safe)."""
    vals = [s.raw_scale for s in samples]
    if not vals:
        return 1.0
    return float(np.percentile(vals, 90))


def _pairs(samples: Sequence[Sample]) -> list[tuple[Sample, Sample]]:
    """Consecutive pairs at least MIN_PAIR_DT_S apart (reduces jitter at high FPS)."""
    pairs: list[tuple[Sample, Sample]] = []
    anchor: Sample | None = None
    for s in samples:
        if anchor is None:
            anchor = s
            continue
        if s.ts - anchor.ts >= MIN_PAIR_DT_S:
            pairs.append((anchor, s))
            anchor = s
    return pairs


def _hip(s: Sample) -> np.ndarray:
    return np.array(s.pose.hip_xy, dtype=np.float64)


def _hip_speed(samples: Sequence[Sample], scale: float, axis: int | None = None) -> float:
    """Net hip displacement over the window / time. Robust to keypoint jitter."""
    if len(samples) < 2:
        return 0.0
    first, last = samples[0], samples[-1]
    dt = last.ts - first.ts
    if dt <= 1e-6:
        return 0.0
    delta = _hip(last) - _hip(first)
    dist = abs(float(delta[axis])) if axis is not None else float(np.linalg.norm(delta))
    return dist / dt / scale


def _vertical_velocity(samples: Sequence[Sample], scale: float) -> float:
    if len(samples) < 2:
        return 0.0
    ts = np.array([s.ts for s in samples])
    ys = np.array([s.pose.hip_xy[1] for s in samples])
    if float(ts[-1] - ts[0]) <= 1e-6:
        return 0.0
    slope = float(np.polyfit(ts - ts[0], ys, 1)[0])
    return slope / scale


def _peak_down_velocity(samples: Sequence[Sample], scale: float) -> float:
    best = 0.0
    for a, b in _pairs(samples):
        v = (b.pose.hip_xy[1] - a.pose.hip_xy[1]) / (b.ts - a.ts) / scale
        best = max(best, v)
    return best


def _visible(s: Sample, *points: int) -> bool:
    return all(s.keypoints.conf[p] >= MIN_POINT_CONF for p in points)


def _gait(samples: Sequence[Sample], scale: float) -> float:
    """Speed of the left-minus-right leg vector. Legs move in anti-phase when walking or
    running, but together when squatting/sitting/falling, so this isolates locomotion."""
    speeds: list[float] = []
    for a, b in _pairs(samples):
        dt = b.ts - a.ts
        for left, right in LEG_PAIRS:
            if not (_visible(a, left, right) and _visible(b, left, right)):
                continue
            da = a.keypoints.xy[left] - a.keypoints.xy[right]
            db = b.keypoints.xy[left] - b.keypoints.xy[right]
            speeds.append(float(np.linalg.norm(db - da)) / dt / scale)
    return float(np.median(speeds)) if speeds else 0.0


def _motion_energy(samples: Sequence[Sample], scale: float) -> float:
    speeds: list[float] = []
    for a, b in _pairs(samples):
        dt = b.ts - a.ts
        ok = (a.keypoints.conf >= MIN_POINT_CONF) & (b.keypoints.conf >= MIN_POINT_CONF)
        if not ok.any():
            continue
        d = np.linalg.norm(b.keypoints.xy[ok] - a.keypoints.xy[ok], axis=1)
        speeds.append(float(np.median(d)) / dt / scale)
    return float(np.median(speeds)) if speeds else 0.0


def _range(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return float(max(vals) - min(vals)) if len(vals) >= 2 else 0.0


def _std(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return float(np.std(vals)) if len(vals) >= 3 else 0.0


def _with_hips(samples: Sequence[Sample]) -> list[Sample]:
    return [s for s in samples if s.pose.hips_visible]


def compute_motion_features(history: TrackHistory) -> MotionFeatures:
    latest = history.latest
    if latest is None:
        raise ValueError("cannot compute motion features on an empty history")

    scale = max(reference_scale(history.window(SCALE_WINDOW_S)), 1.0)
    impact_all = history.window(IMPACT_WINDOW_S)
    tilt_window = history.window(TILT_WINDOW_S)
    osc = history.window(OSCILLATION_WINDOW_S)
    # Hip motion only from frames where the hips were really seen: a box-centre stand-in
    # jumps whenever someone moves partly out of view and would look like a sudden drop.
    impact = _with_hips(impact_all)
    speed_window = _with_hips(history.window(SPEED_WINDOW_S))

    if latest.pose.hips_visible and impact:
        drop = (latest.pose.hip_xy[1] - min(s.pose.hip_xy[1] for s in impact)) / scale
    else:
        drop = 0.0

    tilts = [s.pose.torso_tilt for s in tilt_window if not math.isnan(s.pose.torso_tilt)]
    tilt_now = latest.pose.torso_tilt
    min_tilt = min(tilts) if tilts else math.nan
    tilt_change = (tilt_now - min_tilt) if tilts and not math.isnan(tilt_now) else 0.0

    oldest = next(iter(history))
    return MotionFeatures(
        scale=scale,
        body_speed=_hip_speed(speed_window, scale),
        horizontal_speed=_hip_speed(speed_window, scale, axis=0),
        vertical_velocity=_vertical_velocity(_with_hips(history.window(VERTICAL_WINDOW_S)), scale),
        drop_1s=float(drop),
        peak_down_velocity=_peak_down_velocity(impact, scale),
        tilt_change=float(tilt_change),
        min_recent_tilt=float(min_tilt),
        gait=_gait(history.window(GAIT_WINDOW_S), scale),
        motion_energy=_motion_energy(impact_all, scale),
        elbow_range=_range([s.pose.elbow_angle for s in osc]),
        knee_range=_range([s.pose.knee_angle for s in osc]),
        yaw_std=_std([s.pose.head_yaw for s in osc]),
        span=float(latest.ts - oldest.ts),
    )
