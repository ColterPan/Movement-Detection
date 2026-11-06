"""Record raw camera video to build your own labelled test clips."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from motionsense.sources.opencv_source import CameraSource


def record_clip(source: str, out: Path, seconds: float, width: int, height: int, fps: int) -> int:
    """Record ``seconds`` of video from ``source`` into ``out`` (mp4). Returns frames written."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cam = CameraSource(int(source) if source.isdigit() else source, width, height, fps)
    writer: cv2.VideoWriter | None = None
    written = 0
    start = time.perf_counter()
    next_at = start
    interval = 1.0 / fps
    try:
        for frame in cam.frames():
            now = time.perf_counter()
            if now - start > seconds:
                break
            if now < next_at:  # keep a constant frame rate in the file
                continue
            next_at += interval
            if writer is None:
                h, w = frame.image.shape[:2]
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame.image)
            written += 1
    finally:
        cam.close()
        if writer is not None:
            writer.release()
    return written
