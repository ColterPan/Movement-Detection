"""Synthetic source + scripted perception backends driven by a Scenario."""

from __future__ import annotations

import time
from collections.abc import Iterator

import cv2
import numpy as np

from motionsense.simulation.scenarios import FRAME_H, FRAME_W, Scenario
from motionsense.simulation.skeleton import build_skeleton
from motionsense.types import Frame, ObjectDetection, PersonDetection

BACKGROUND = (38, 32, 28)
FLOOR = (70, 62, 55)
FURNITURE = (90, 80, 70)


def render_background(scenario: Scenario, size: tuple[int, int] = (FRAME_W, FRAME_H)) -> np.ndarray:
    w, h = size
    img = np.full((h, w, 3), BACKGROUND, dtype=np.uint8)
    floor_y = int(0.92 * h)
    cv2.rectangle(img, (0, floor_y), (w, h), FLOOR, -1)
    for obj in scenario.objects:
        x1, y1, x2, y2 = (int(v) for v in obj.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), FURNITURE, -1)
        cv2.putText(
            img,
            obj.label,
            (x1 + 8, y1 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 190, 180),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        img,
        f"SIMULATION: {scenario.name}",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (120, 200, 255),
        2,
        cv2.LINE_AA,
    )
    return img


class SyntheticSource:
    """Produces frames with media-time timestamps for a scenario (optionally looping)."""

    def __init__(
        self, scenario: Scenario, fps: float = 15.0, realtime: bool = False, loop: bool = False
    ) -> None:
        self.scenario = scenario
        self.fps = fps
        self.is_live = realtime
        self._realtime = realtime
        self._loop = loop
        self._closed = False
        self.description = f"synthetic:{scenario.name}"
        self._background = render_background(scenario)

    def frames(self) -> Iterator[Frame]:
        index = 0
        start = time.perf_counter()
        total = int(self.scenario.duration * self.fps)
        while not self._closed and (self._loop or index < total):
            ts = index / self.fps
            if self._realtime:
                delay = ts - (time.perf_counter() - start)
                if delay > 0:
                    time.sleep(delay)
            yield Frame(image=self._background.copy(), ts=ts, index=index)
            index += 1

    def close(self) -> None:
        self._closed = True


class ScriptedPoseBackend:
    """Returns the scenario's skeleton for the frame's timestamp (with optional noise)."""

    name = "scripted-pose"

    def __init__(
        self, scenario: Scenario, noise_px: float = 0.0, seed: int = 0, loop: bool = False
    ) -> None:
        self._scenario = scenario
        self._noise = noise_px
        self._rng = np.random.default_rng(seed)
        self._loop = loop

    def infer(self, frame: Frame) -> list[PersonDetection]:
        t = frame.ts % self._scenario.duration if self._loop else frame.ts
        params = self._scenario.pose_at(t)
        if params.visible < 0.5:
            return []
        kps, bbox = build_skeleton(params, frame.size, rng=self._rng, noise_px=self._noise)
        return [PersonDetection(bbox=bbox, score=0.95, keypoints=kps, track_id=1)]


class ScriptedObjectBackend:
    name = "scripted-objects"

    def __init__(self, scenario: Scenario) -> None:
        self._objects = list(scenario.objects)

    def infer(self, frame: Frame) -> list[ObjectDetection]:
        return list(self._objects)
