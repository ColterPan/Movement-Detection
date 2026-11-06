"""Ultralytics YOLO pose backend with built-in ByteTrack tracking."""

from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from motionsense.config import ModelConfig
from motionsense.perception.model_manager import resolve_device, resolve_model, warm_up
from motionsense.types import Frame, Keypoints, PersonDetection

logger = logging.getLogger(__name__)

TRACKER_INPUT_CONF = 0.1
DEFAULT_TRACKER = "motionsense"


def resolve_tracker(name: str) -> str:
    """``motionsense`` = the tuned ByteTrack config shipped with this package."""
    if name == DEFAULT_TRACKER:
        return str(resources.files("motionsense.perception") / "bytetrack_motionsense.yaml")
    return name


def _to_numpy(t: Any) -> np.ndarray:
    return t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)


class YoloPoseBackend:
    name = "yolo-pose"

    def __init__(self, cfg: ModelConfig, models_dir: Path) -> None:
        self._cfg = cfg
        self.device = resolve_device(cfg.runtime, cfg.device)
        self._model = resolve_model(cfg.pose_model, models_dir, cfg.runtime, cfg.imgsz, task="pose")
        warm_up(self._model, self.device, cfg.imgsz, "pose")
        self._fresh_tracker = True
        self._tracker = resolve_tracker(cfg.tracker)
        logger.info("pose model %s ready on %s", cfg.pose_model, self.device)

    def reset(self) -> None:
        """Forget all tracks (e.g. before processing a different video)."""
        self._fresh_tracker = True

    def infer(self, frame: Frame) -> list[PersonDetection]:
        persist = not self._fresh_tracker  # persist=False makes Ultralytics build a new tracker
        self._fresh_tracker = False
        result = self._model.track(
            frame.image,
            persist=persist,
            tracker=self._tracker,
            imgsz=self._cfg.imgsz,
            # ByteTrack needs the low-score boxes too: they keep existing tracks alive.
            # New tracks still require pose_conf (see the tracker config / pipeline).
            conf=TRACKER_INPUT_CONF,
            device=self.device,
            verbose=False,
        )[0]
        return parse_pose_result(result, min_untracked_score=self._cfg.pose_conf)


def parse_pose_result(result: Any, min_untracked_score: float = 0.0) -> list[PersonDetection]:
    """Convert an Ultralytics result. Boxes without a track id (the tracker returned nothing
    for them) must still reach ``min_untracked_score`` to count as a person."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0 or result.keypoints is None:
        return []
    xyxy = _to_numpy(boxes.xyxy)
    scores = _to_numpy(boxes.conf)
    ids = _to_numpy(boxes.id).astype(int) if boxes.id is not None else None
    kxy = _to_numpy(result.keypoints.xy)
    kconf = (
        _to_numpy(result.keypoints.conf)
        if result.keypoints.conf is not None
        else np.ones(kxy.shape[:2])
    )
    people: list[PersonDetection] = []
    for i in range(len(xyxy)):
        if ids is None and float(scores[i]) < min_untracked_score:
            continue
        x1, y1, x2, y2 = (float(v) for v in xyxy[i])
        people.append(
            PersonDetection(
                bbox=(x1, y1, x2, y2),
                score=float(scores[i]),
                keypoints=Keypoints(xy=kxy[i], conf=kconf[i]),
                track_id=int(ids[i]) if ids is not None else None,
            )
        )
    return people
