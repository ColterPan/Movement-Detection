"""OpenCV-backed sources: webcams, network streams (RTSP/HTTP) and video files."""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np

from motionsense.types import Frame

logger = logging.getLogger(__name__)

WARMUP_FRAMES = 8  # webcams return dark frames while auto-exposure settles
RECONNECT_DELAY_S = 2.0
READ_TIMEOUT_S = 1.0


def redact_url(spec: str | int) -> str:
    """Hide credentials in stream URLs (rtsp://user:pass@host -> rtsp://***@host)."""
    text = str(spec)
    parts = urlsplit(text)
    if not parts.scheme or "@" not in parts.netloc:
        return text
    host = parts.netloc.rsplit("@", 1)[1]
    return urlunsplit(parts._replace(netloc=f"***@{host}"))


def _open_capture(spec: str | int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    if isinstance(spec, int) and sys.platform == "win32":
        cap = cv2.VideoCapture(spec, cv2.CAP_DSHOW)  # opens much faster than MSMF on Windows
    else:
        cap = cv2.VideoCapture(spec)
    if isinstance(spec, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class CameraSource:
    """Live camera or network stream read on a background thread.

    Only the newest frame is kept, so slow processing never builds up latency.
    Reconnects automatically if the stream drops.
    """

    is_live = True

    def __init__(
        self, spec: str | int, width: int = 1280, height: int = 720, fps: int = 30
    ) -> None:
        self._spec = spec
        self._size = (width, height, fps)
        self._safe_name = redact_url(spec)
        self.description = f"camera:{self._safe_name}"
        self._cond = threading.Condition()
        self._latest: np.ndarray | None = None
        self._latest_index = -1
        self._closed = threading.Event()
        self.error: str | None = None
        self._thread = threading.Thread(target=self._reader, name="camera-reader", daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        index = 0
        while not self._closed.is_set():
            cap = _open_capture(self._spec, *self._size)
            if not cap.isOpened():
                self.error = f"cannot open camera/stream {self._safe_name}"
                logger.warning("%s; retrying in %.0fs", self.error, RECONNECT_DELAY_S)
                cap.release()
                self._closed.wait(RECONNECT_DELAY_S)
                continue
            self.error = None
            for _ in range(WARMUP_FRAMES):
                cap.read()
            while not self._closed.is_set():
                ok, img = cap.read()
                if not ok or img is None:
                    self.error = f"stream {self._safe_name} stopped delivering frames"
                    logger.warning("%s; reconnecting", self.error)
                    break
                with self._cond:
                    self._latest = img
                    self._latest_index = index
                    self._cond.notify_all()
                index += 1
            cap.release()
            self._closed.wait(RECONNECT_DELAY_S)

    def frames(self) -> Iterator[Frame]:
        last = -1
        while not self._closed.is_set():
            with self._cond:
                self._cond.wait_for(partial(self._has_news, last), timeout=READ_TIMEOUT_S)
                if self._latest is None or self._latest_index == last:
                    continue
                img, last = self._latest, self._latest_index
            yield Frame(image=img, ts=time.time(), index=last)

    def _has_news(self, last: int) -> bool:
        return self._latest_index != last or self._closed.is_set()

    def close(self) -> None:
        self._closed.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=3.0)


class VideoFileSource:
    """Reads every frame of a video file. Timestamps are media time, so results are
    reproducible regardless of processing speed (used for evaluation)."""

    is_live = False

    def __init__(self, path: str | Path, realtime: bool = False) -> None:
        self._path = Path(path)
        if not self._path.is_file():
            raise FileNotFoundError(f"video not found: {self._path}")
        self._realtime = realtime
        self._closed = False
        self.description = f"file:{self._path.name}"
        cap = cv2.VideoCapture(str(self._path))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    def frames(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(str(self._path))
        start = time.perf_counter()
        index = 0
        try:
            while not self._closed:
                ok, img = cap.read()
                if not ok:
                    return
                ts = index / self.fps
                if self._realtime:
                    delay = ts - (time.perf_counter() - start)
                    if delay > 0:
                        time.sleep(delay)
                yield Frame(image=np.asarray(img, dtype=np.uint8), ts=ts, index=index)
                index += 1
        finally:
            cap.release()

    def close(self) -> None:
        self._closed = True
