"""Perception backend interfaces. Implement these to plug in MediaPipe, RTMPose, etc."""

from __future__ import annotations

from typing import Protocol

from motionsense.types import Frame, ObjectDetection, PersonDetection


class PoseBackend(Protocol):
    """Detects people and their keypoints. Should return stable ``track_id`` values when it
    tracks; the pipeline assigns transient negative ids otherwise."""

    name: str

    def infer(self, frame: Frame) -> list[PersonDetection]: ...


class ObjectBackend(Protocol):
    name: str

    def infer(self, frame: Frame) -> list[ObjectDetection]: ...
