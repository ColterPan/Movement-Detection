"""Core immutable value types passed between pipeline stages.

All coordinates are in pixels of the source frame, origin top-left, y pointing down.
Keypoints follow the COCO-17 layout (see ``KP``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BBox = tuple[float, float, float, float]  # x1, y1, x2, y2


class KP(IntEnum):
    """COCO-17 keypoint indices."""

    NOSE = 0
    L_EYE = 1
    R_EYE = 2
    L_EAR = 3
    R_EAR = 4
    L_SHOULDER = 5
    R_SHOULDER = 6
    L_ELBOW = 7
    R_ELBOW = 8
    L_WRIST = 9
    R_WRIST = 10
    L_HIP = 11
    R_HIP = 12
    L_KNEE = 13
    R_KNEE = 14
    L_ANKLE = 15
    R_ANKLE = 16


NUM_KEYPOINTS = 17

SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    (KP.L_SHOULDER, KP.R_SHOULDER),
    (KP.L_SHOULDER, KP.L_ELBOW),
    (KP.L_ELBOW, KP.L_WRIST),
    (KP.R_SHOULDER, KP.R_ELBOW),
    (KP.R_ELBOW, KP.R_WRIST),
    (KP.L_SHOULDER, KP.L_HIP),
    (KP.R_SHOULDER, KP.R_HIP),
    (KP.L_HIP, KP.R_HIP),
    (KP.L_HIP, KP.L_KNEE),
    (KP.L_KNEE, KP.L_ANKLE),
    (KP.R_HIP, KP.R_KNEE),
    (KP.R_KNEE, KP.R_ANKLE),
    (KP.NOSE, KP.L_EYE),
    (KP.NOSE, KP.R_EYE),
    (KP.L_EYE, KP.L_EAR),
    (KP.R_EYE, KP.R_EAR),
)


def _readonly(arr: NDArray[Any]) -> NDArray[Any]:
    out = np.array(arr, dtype=np.float64, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True, slots=True)
class Keypoints:
    """17 keypoints with per-point confidence. Arrays are read-only."""

    xy: FloatArray  # shape (17, 2)
    conf: FloatArray  # shape (17,)

    def __post_init__(self) -> None:
        xy = np.asarray(self.xy, dtype=np.float64)
        conf = np.asarray(self.conf, dtype=np.float64)
        if xy.shape != (NUM_KEYPOINTS, 2):
            raise ValueError(f"keypoints xy must have shape (17, 2), got {xy.shape}")
        if conf.shape != (NUM_KEYPOINTS,):
            raise ValueError(f"keypoint conf must have shape (17,), got {conf.shape}")
        object.__setattr__(self, "xy", _readonly(xy))
        object.__setattr__(self, "conf", _readonly(conf))

    def point(self, idx: int) -> FloatArray:
        return self.xy[idx]

    def visible(self, idx: int, min_conf: float) -> bool:
        return bool(self.conf[idx] >= min_conf)


@dataclass(frozen=True, slots=True)
class Frame:
    """A single video frame. ``ts`` is seconds (wall clock for live, media time for files)."""

    image: NDArray[np.uint8]
    ts: float
    index: int

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return w, h


@dataclass(frozen=True, slots=True)
class PersonDetection:
    """Raw output of a pose backend for one person."""

    bbox: BBox
    score: float
    keypoints: Keypoints
    track_id: int | None = None


@dataclass(frozen=True, slots=True)
class ObjectDetection:
    label: str
    score: float
    bbox: BBox


@dataclass(frozen=True, slots=True)
class PersonState:
    """Fully analysed state of one tracked person at one instant."""

    track_id: int
    bbox: BBox
    keypoints: Keypoints
    activities: dict[str, float]
    top_activity: str
    top_confidence: float
    quality: str  # "full" | "partial" | "low"
    fall_state: str
    fall_risk: float
    reps: dict[str, int] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)
    support: str | None = None  # furniture under the hips, if any
    features: dict[str, float] = field(default_factory=dict)

    def to_dict(self, include_keypoints: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "track_id": self.track_id,
            "bbox": [round(v, 1) for v in self.bbox],
            "activities": {k: round(v, 4) for k, v in self.activities.items()},
            "top_activity": self.top_activity,
            "top_confidence": round(self.top_confidence, 4),
            "quality": self.quality,
            "fall_state": self.fall_state,
            "fall_risk": round(self.fall_risk, 4),
            "reps": dict(self.reps),
            "attributes": dict(self.attributes),
            "support": self.support,
            "features": {k: round(v, 3) for k, v in self.features.items()},
        }
        if include_keypoints:
            data["keypoints"] = [
                [round(float(x), 1), round(float(y), 1), round(float(c), 3)]
                for (x, y), c in zip(self.keypoints.xy, self.keypoints.conf, strict=True)
            ]
        return data


@dataclass(frozen=True, slots=True)
class SceneState:
    """Everything known about the scene at one instant. This is what the API serves."""

    ts: float
    frame_index: int
    frame_size: tuple[int, int]
    fps: float
    persons: tuple[PersonState, ...]
    objects: tuple[ObjectDetection, ...]

    def to_dict(self, include_keypoints: bool = True) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "frame_index": self.frame_index,
            "frame_size": list(self.frame_size),
            "fps": round(self.fps, 2),
            "person_count": len(self.persons),
            "persons": [p.to_dict(include_keypoints) for p in self.persons],
            "objects": [
                {
                    "label": o.label,
                    "score": round(o.score, 3),
                    "bbox": [round(v, 1) for v in o.bbox],
                }
                for o in self.objects
            ],
        }


EMPTY_SCENE = SceneState(ts=0.0, frame_index=-1, frame_size=(0, 0), fps=0.0, persons=(), objects=())
