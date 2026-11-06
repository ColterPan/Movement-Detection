"""Per-frame (static) pose geometry.

Everything here is a pure function of one set of keypoints. Distances are later normalised
by the torso length, so features are independent of how far the person is from the camera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from motionsense.types import KP, BBox, FloatArray, Keypoints

NAN = float("nan")


@dataclass(frozen=True, slots=True)
class PoseFeatures:
    """Static, single-frame description of a pose. Angles in degrees, NaN when unknown."""

    torso_tilt: float  # 0 = upright, 90 = horizontal (folded into 0..90)
    torso_len: float  # px, shoulder-mid to hip-mid
    hip_angle: float  # shoulder-hip-knee, 180 = straight
    knee_angle: float  # hip-knee-ankle, 180 = straight
    elbow_angle: float  # shoulder-elbow-wrist
    bbox_aspect: float  # width / height
    head_yaw: float  # approx -1 (looking left) .. +1 (looking right), NaN when unknown
    wrist_support: float  # (wrist_y - shoulder_y) / torso_len; > 0.5 means arms pushing on floor
    hip_xy: tuple[float, float]
    head_y: float
    has_torso: bool
    has_legs: bool
    quality: float  # mean keypoint confidence
    posture_tilt: float = NAN  # best tilt estimate: torso, else head-over-shoulders, else box shape
    hips_visible: bool = False  # False: hip_xy is only the box centre (unusable for motion)
    truncated: bool = False  # the person's box touches the frame edge (partly out of view)


def midpoint(kps: Keypoints, a: int, b: int, min_conf: float) -> FloatArray | None:
    """Midpoint of two keypoints, or the single visible one, or None."""
    va, vb = kps.visible(a, min_conf), kps.visible(b, min_conf)
    if va and vb:
        return (kps.xy[a] + kps.xy[b]) / 2.0
    if va:
        return kps.xy[a]
    if vb:
        return kps.xy[b]
    return None


def joint_angle(a: FloatArray, b: FloatArray, c: FloatArray) -> float:
    """Interior angle ABC in degrees (180 = straight line)."""
    v1 = a - b
    v2 = c - b
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return NAN
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def tilt_from_vertical(top: FloatArray, bottom: FloatArray) -> float:
    """Angle of the segment bottom->top from vertical, folded into [0, 90]."""
    dx = float(top[0] - bottom[0])
    dy = float(bottom[1] - top[1])  # positive when top is above bottom
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return NAN
    angle = math.degrees(math.atan2(abs(dx), dy))  # 0..180
    return angle if angle <= 90.0 else 180.0 - angle


def _side_angle(kps: Keypoints, a: int, b: int, c: int, min_conf: float) -> float:
    if kps.visible(a, min_conf) and kps.visible(b, min_conf) and kps.visible(c, min_conf):
        return joint_angle(kps.xy[a], kps.xy[b], kps.xy[c])
    return NAN


def _mean_angle(*angles: float) -> float:
    vals = [a for a in angles if not math.isnan(a)]
    return float(sum(vals) / len(vals)) if vals else NAN


def _min_angle(*angles: float) -> float:
    vals = [a for a in angles if not math.isnan(a)]
    return float(min(vals)) if vals else NAN


def head_yaw(kps: Keypoints, min_conf: float) -> float:
    """Rough head yaw from nose position relative to the ears (or eyes)."""
    if not kps.visible(KP.NOSE, min_conf):
        return NAN
    nose_x = float(kps.xy[KP.NOSE][0])
    for left, right in ((KP.L_EAR, KP.R_EAR), (KP.L_EYE, KP.R_EYE)):
        lv, rv = kps.visible(left, min_conf), kps.visible(right, min_conf)
        if lv and rv:
            lx, rx = float(kps.xy[left][0]), float(kps.xy[right][0])
            half = abs(lx - rx) / 2.0
            if half < 1e-6:
                continue
            centre = (lx + rx) / 2.0
            return float(np.clip((nose_x - centre) / half, -1.0, 1.0))
    # Only one ear visible: the head is turned towards the hidden side.
    if kps.visible(KP.L_EAR, min_conf) != kps.visible(KP.R_EAR, min_conf):
        ear = KP.L_EAR if kps.visible(KP.L_EAR, min_conf) else KP.R_EAR
        return 1.0 if nose_x > float(kps.xy[ear][0]) else -1.0
    return NAN


def bbox_aspect(bbox: BBox) -> float:
    w = max(bbox[2] - bbox[0], 1e-6)
    h = max(bbox[3] - bbox[1], 1e-6)
    return float(w / h)


EDGE_FRACTION = 0.015  # keypoints this close to the frame border are not trusted


def _edge_margin(frame_size: tuple[int, int]) -> tuple[float, float]:
    w, h = frame_size
    return max(3.0, EDGE_FRACTION * w), max(3.0, EDGE_FRACTION * h)


def is_truncated(bbox: BBox, frame_size: tuple[int, int] | None) -> bool:
    """True when the box touches the left, right or bottom edge of the frame."""
    if frame_size is None:
        return False
    mx, my = _edge_margin(frame_size)
    w, h = frame_size
    return bbox[0] <= mx or bbox[2] >= w - mx or bbox[3] >= h - my


def drop_edge_keypoints(kps: Keypoints, frame_size: tuple[int, int] | None) -> Keypoints:
    """Pose models *guess* body parts that are outside the image and pile them up on the border.
    Treat keypoints on (or beyond) the left/right/bottom edge as not visible."""
    if frame_size is None:
        return kps
    mx, my = _edge_margin(frame_size)
    w, h = frame_size
    x, y = kps.xy[:, 0], kps.xy[:, 1]
    on_edge = (x <= mx) | (x >= w - mx) | (y >= h - my)
    if not on_edge.any():
        return kps
    return Keypoints(xy=kps.xy, conf=np.where(on_edge, 0.0, kps.conf))


def _head_tilt(kps: Keypoints, sh: FloatArray | None, min_conf: float) -> float:
    """Tilt of the shoulders->head line: ~0 when the head is above the shoulders."""
    head = midpoint(kps, KP.L_EYE, KP.R_EYE, min_conf)
    if head is None and kps.visible(KP.NOSE, min_conf):
        head = kps.xy[KP.NOSE]
    if head is None or sh is None:
        return NAN
    return tilt_from_vertical(head, sh)


def compute_pose_features(
    kps: Keypoints,
    bbox: BBox,
    min_conf: float = 0.35,
    frame_size: tuple[int, int] | None = None,
) -> PoseFeatures:
    """Compute static pose features for one person.

    Pass ``frame_size`` so keypoints guessed outside the image are ignored and partly
    visible people are recognised as such.
    """
    kps = drop_edge_keypoints(kps, frame_size)
    truncated = is_truncated(bbox, frame_size)
    sh = midpoint(kps, KP.L_SHOULDER, KP.R_SHOULDER, min_conf)
    hip = midpoint(kps, KP.L_HIP, KP.R_HIP, min_conf)
    has_torso = sh is not None and hip is not None

    torso_len = float(np.linalg.norm(sh - hip)) if has_torso else NAN  # type: ignore[operator]
    tilt = tilt_from_vertical(sh, hip) if has_torso else NAN  # type: ignore[arg-type]

    hip_angle = _mean_angle(
        _side_angle(kps, KP.L_SHOULDER, KP.L_HIP, KP.L_KNEE, min_conf),
        _side_angle(kps, KP.R_SHOULDER, KP.R_HIP, KP.R_KNEE, min_conf),
    )
    knee_angle = _mean_angle(
        _side_angle(kps, KP.L_HIP, KP.L_KNEE, KP.L_ANKLE, min_conf),
        _side_angle(kps, KP.R_HIP, KP.R_KNEE, KP.R_ANKLE, min_conf),
    )
    elbow_angle = _min_angle(
        _side_angle(kps, KP.L_SHOULDER, KP.L_ELBOW, KP.L_WRIST, min_conf),
        _side_angle(kps, KP.R_SHOULDER, KP.R_ELBOW, KP.R_WRIST, min_conf),
    )
    has_legs = not math.isnan(hip_angle) or not math.isnan(knee_angle)

    wrist = midpoint(kps, KP.L_WRIST, KP.R_WRIST, min_conf)
    if wrist is not None and sh is not None and has_torso and torso_len > 1e-6:
        wrist_support = float((wrist[1] - sh[1]) / torso_len)
    else:
        wrist_support = NAN

    if hip is not None:
        hip_xy = (float(hip[0]), float(hip[1]))
    else:
        hip_xy = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    if has_torso:
        posture_tilt = tilt
    elif not math.isnan(head_tilt := _head_tilt(kps, sh, min_conf)):
        posture_tilt = head_tilt
    elif not truncated:
        # Box shape only means something when the whole body is in the frame.
        posture_tilt = max(0.0, min(90.0, (bbox_aspect(bbox) - 0.4) / 1.6 * 90.0))
    else:
        posture_tilt = NAN

    if kps.visible(KP.NOSE, min_conf):
        head_y = float(kps.xy[KP.NOSE][1])
    elif sh is not None:
        head_y = float(sh[1])
    else:
        head_y = float(bbox[1])

    return PoseFeatures(
        torso_tilt=tilt,
        torso_len=torso_len,
        hip_angle=hip_angle,
        knee_angle=knee_angle,
        elbow_angle=elbow_angle,
        bbox_aspect=bbox_aspect(bbox),
        head_yaw=head_yaw(kps, min_conf),
        wrist_support=wrist_support,
        hip_xy=hip_xy,
        head_y=head_y,
        has_torso=has_torso,
        has_legs=has_legs,
        quality=float(np.mean(kps.conf)),
        posture_tilt=posture_tilt,
        hips_visible=hip is not None,
        truncated=truncated,
    )
