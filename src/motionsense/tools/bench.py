"""Measure end-to-end throughput of the perception models on this machine."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from motionsense.config import Settings
from motionsense.perception.yolo_objects import YoloObjectBackend
from motionsense.perception.yolo_pose import YoloPoseBackend
from motionsense.sources.opencv_source import CameraSource, VideoFileSource
from motionsense.types import Frame

WARMUP = 5


@dataclass(frozen=True, slots=True)
class BenchResult:
    frames: int
    pose_ms: float
    objects_ms: float | None
    pose_fps: float
    people_seen: int
    device: str


def _time(fn: Callable[[Frame], object], frame: Frame) -> float:
    t = time.perf_counter()
    fn(frame)
    return (time.perf_counter() - t) * 1000.0


def run_bench(settings: Settings, seconds: float = 15.0) -> BenchResult:
    spec = settings.camera.source
    src = (
        CameraSource(int(spec), settings.camera.width, settings.camera.height, settings.camera.fps)
        if spec.isdigit()
        else VideoFileSource(spec)
    )
    pose = YoloPoseBackend(settings.models, settings.models_dir)
    objects = (
        YoloObjectBackend(settings.models, settings.models_dir)
        if settings.objects.enabled
        else None
    )
    pose_times: list[float] = []
    obj_times: list[float] = []
    people = 0
    end = None
    try:
        for i, frame in enumerate(src.frames()):
            if i < WARMUP:
                pose.infer(frame)
                continue
            end = end or time.perf_counter() + seconds
            t = time.perf_counter()
            people = max(people, len(pose.infer(frame)))
            pose_times.append((time.perf_counter() - t) * 1000.0)
            if objects is not None and i % 10 == 0:
                obj_times.append(_time(objects.infer, frame))
            if time.perf_counter() > end:
                break
    finally:
        src.close()
    pose_ms = sum(pose_times) / max(len(pose_times), 1)
    return BenchResult(
        frames=len(pose_times),
        pose_ms=round(pose_ms, 1),
        objects_ms=round(sum(obj_times) / len(obj_times), 1) if obj_times else None,
        pose_fps=round(1000.0 / pose_ms, 1) if pose_ms else 0.0,
        people_seen=people,
        device=pose.device,
    )
