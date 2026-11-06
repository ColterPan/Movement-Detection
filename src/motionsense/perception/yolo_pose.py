"""Ultralytics YOLO pose backend with built-in ByteTrack tracking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from motionsense.config import ModelConfig
from motionsense.perception.model_manager import resolve_device, resolve_model, warm_up
from motionsense.types import Frame, Keypoints, PersonDetection

logger = logging.getLogger(__name__)


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
            tracker=self._cfg.tracker,
            imgsz=self._cfg.imgsz,
            conf=self._cfg.pose_conf,
            device=self.device,
            verbose=False,
        )[0]
        return parse_pose_result(result)


def parse_pose_result(result: Any) -> list[PersonDetection]:
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
