"""Procedural 2D skeleton generator (COCO-17) from a handful of joint angles.

Used for deterministic tests, accuracy regression checks and the camera-free demo mode.
Conventions: the figure faces +x; ``tilt`` > 0 leans the torso forward (90 = lying, head
towards +x); angles are interior joint angles (180 = straight).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace

import numpy as np

from motionsense.types import KP, BBox, Keypoints

THIGH = 0.9
SHIN = 0.85
UPPER_ARM = 0.6
FOREARM = 0.55


@dataclass(frozen=True, slots=True)
class PoseParams:
    x: float = 0.5  # hip x as a fraction of frame width
    tilt: float = 0.0
    hip: float = 178.0
    knee: float = 178.0
    shoulder: float = 12.0  # arm angle away from the torso-down direction, forwards
    elbow: float = 165.0
    leg_swing: float = 0.0  # deg; + = left leg forwards, right leg backwards
    yaw: float = 0.0  # -1..1 head turn
    lift: float = 0.0  # raise the whole body by this many torso lengths (bed, chair seat)
    visible: float = 1.0  # 0 = person not in view

    def lerp(self, other: PoseParams, a: float) -> PoseParams:
        values = {
            f.name: getattr(self, f.name) + (getattr(other, f.name) - getattr(self, f.name)) * a
            for f in fields(self)
        }
        return PoseParams(**values)

    def with_(self, **changes: float) -> PoseParams:
        return replace(self, **changes)


STAND = PoseParams()
SIT = PoseParams(tilt=5.0, hip=95.0, knee=95.0, shoulder=25.0, elbow=110.0)
LIE = PoseParams(tilt=90.0, hip=175.0, knee=170.0, shoulder=10.0, elbow=170.0)
PLANK = PoseParams(tilt=78.0, hip=178.0, knee=178.0, shoulder=78.0, elbow=175.0)
SQUAT = PoseParams(tilt=30.0, hip=75.0, knee=70.0, shoulder=80.0, elbow=170.0)


def _rot(v: np.ndarray, deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c])


def build_skeleton(
    p: PoseParams,
    frame_size: tuple[int, int] = (1280, 720),
    torso_frac: float = 0.18,
    floor_frac: float = 0.92,
    rng: np.random.Generator | None = None,
    noise_px: float = 0.0,
) -> tuple[Keypoints, BBox]:
    """Return keypoints and a bounding box for the given pose parameters."""
    w, h = frame_size
    unit = torso_frac * h
    t = math.radians(p.tilt)
    up = np.array([math.sin(t), -math.cos(t)])  # hip -> shoulder
    side = np.array([math.cos(t), math.sin(t)])  # across the body
    down = -up

    pts = np.zeros((17, 2))
    hip_c = np.zeros(2)
    sh_c = hip_c + unit * up
    pts[KP.L_HIP], pts[KP.R_HIP] = hip_c - 0.15 * unit * side, hip_c + 0.15 * unit * side
    pts[KP.L_SHOULDER], pts[KP.R_SHOULDER] = sh_c - 0.22 * unit * side, sh_c + 0.22 * unit * side

    head = sh_c + 0.42 * unit * up
    pts[KP.L_EAR], pts[KP.R_EAR] = head - 0.1 * unit * side, head + 0.1 * unit * side
    pts[KP.NOSE] = head + 0.05 * unit * up + p.yaw * 0.1 * unit * side
    pts[KP.L_EYE] = head + 0.08 * unit * up - 0.05 * unit * side + p.yaw * 0.06 * unit * side
    pts[KP.R_EYE] = head + 0.08 * unit * up + 0.05 * unit * side + p.yaw * 0.06 * unit * side

    for hip_idx, knee_idx, ankle_idx, swing in (
        (KP.L_HIP, KP.L_KNEE, KP.L_ANKLE, p.leg_swing),
        (KP.R_HIP, KP.R_KNEE, KP.R_ANKLE, -p.leg_swing),
    ):
        thigh = _rot(down, -(180.0 - p.hip) - swing)
        shin = _rot(thigh, 180.0 - p.knee)
        pts[knee_idx] = pts[hip_idx] + THIGH * unit * thigh
        pts[ankle_idx] = pts[knee_idx] + SHIN * unit * shin

    upper = _rot(down, -p.shoulder)
    fore = _rot(upper, -(180.0 - p.elbow))
    for sh_idx, el_idx, wr_idx in (
        (KP.L_SHOULDER, KP.L_ELBOW, KP.L_WRIST),
        (KP.R_SHOULDER, KP.R_ELBOW, KP.R_WRIST),
    ):
        pts[el_idx] = pts[sh_idx] + UPPER_ARM * unit * upper
        pts[wr_idx] = pts[el_idx] + FOREARM * unit * fore

    # Ground the figure: lowest point on the floor (or on the raised surface).
    floor_y = floor_frac * h - p.lift * unit
    pts[:, 1] += floor_y - pts[:, 1].max()
    pts[:, 0] += p.x * w

    if rng is not None and noise_px > 0:
        pts = pts + rng.normal(0.0, noise_px, pts.shape)

    conf = np.full(17, 0.9)
    pad = 0.12 * unit
    bbox = (
        float(pts[:, 0].min() - pad),
        float(pts[:, 1].min() - pad),
        float(pts[:, 0].max() + pad),
        float(pts[:, 1].max() + pad),
    )
    return Keypoints(xy=pts, conf=conf), bbox
