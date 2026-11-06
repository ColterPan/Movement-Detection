"""Ultralytics YOLO object detector (COCO-80 classes)."""

from __future__ import annotations

from pathlib import Path

from motionsense.config import ModelConfig
from motionsense.perception.model_manager import resolve_device, resolve_model, warm_up
from motionsense.perception.yolo_pose import _to_numpy
from motionsense.types import Frame, ObjectDetection

IGNORED_LABELS = frozenset({"person"})  # people come from the pose backend


class YoloObjectBackend:
    name = "yolo-objects"

    def __init__(self, cfg: ModelConfig, models_dir: Path) -> None:
        self._cfg = cfg
        self.device = resolve_device(cfg.runtime, cfg.device)
        self._model = resolve_model(
            cfg.object_model, models_dir, cfg.runtime, cfg.imgsz, task="detect"
        )
        warm_up(self._model, self.device, cfg.imgsz, "detect")

    def infer(self, frame: Frame) -> list[ObjectDetection]:
        result = self._model.predict(
            frame.image,
            imgsz=self._cfg.imgsz,
            conf=self._cfg.object_conf,
            device=self.device,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        names = result.names
        xyxy = _to_numpy(result.boxes.xyxy)
        cls = _to_numpy(result.boxes.cls).astype(int)
        conf = _to_numpy(result.boxes.conf)
        out: list[ObjectDetection] = []
        for box, c, s in zip(xyxy, cls, conf, strict=True):
            label = str(names[int(c)])
            if label in IGNORED_LABELS:
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            out.append(ObjectDetection(label=label, score=float(s), bbox=(x1, y1, x2, y2)))
        return out
