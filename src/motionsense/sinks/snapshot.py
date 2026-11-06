"""Save before/during/after frames for incident events.

These snapshot sequences are what a human (or a vision LLM, in phase 2) reviews to verify
an incident. Raw frames are kept in a small in-memory ring buffer; nothing is written to
disk unless an incident happens.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from motionsense.events.models import INCIDENT_TYPES, Event
from motionsense.render.overlay import encode_jpeg
from motionsense.types import Frame, SceneState

logger = logging.getLogger(__name__)

_SAFE_ID = re.compile(r"^[0-9a-f]{32}$")


def is_safe_event_id(event_id: str) -> bool:
    return bool(_SAFE_ID.match(event_id))


@dataclass
class _Pending:
    event: Event
    folder: Path
    until: float
    next_at: float
    count: int


class SnapshotRecorder:
    def __init__(
        self,
        root: Path,
        pre_s: float = 2.0,
        post_s: float = 2.0,
        hz: float = 2.0,
        jpeg_quality: int = 85,
        max_events: int = 200,
    ) -> None:
        self.root = root
        self._pre_s = pre_s
        self._post_s = post_s
        self._interval = 1.0 / hz
        self._max_events = max_events
        self._quality = jpeg_quality
        self._ring: deque[tuple[float, bytes]] = deque()
        self._last_sample = float("-inf")
        self._latest_ts = 0.0
        self._pending: list[_Pending] = []
        self._lock = threading.Lock()

    # Called by the pipeline for every processed frame.
    def on_frame(self, frame: Frame, scene: SceneState) -> None:
        self._latest_ts = frame.ts
        due_ring = frame.ts - self._last_sample >= self._interval
        with self._lock:
            pending_due = [p for p in self._pending if frame.ts >= p.next_at]
        if not due_ring and not pending_due:
            return
        jpeg = encode_jpeg(np.asarray(frame.image), self._quality)
        if due_ring:
            self._last_sample = frame.ts
            self._ring.append((frame.ts, jpeg))
            while self._ring and self._ring[0][0] < frame.ts - self._pre_s:
                self._ring.popleft()
        for p in pending_due:
            self._write(p.folder, f"post_{p.count:02d}.jpg", jpeg)
            p.count += 1
            p.next_at = frame.ts + self._interval
        with self._lock:
            self._pending = [p for p in self._pending if frame.ts < p.until]

    # Subscribed to the event bus.
    def __call__(self, event: Event) -> None:
        if event.type not in INCIDENT_TYPES:
            return
        self._prune()
        folder = self.root / event.id
        try:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "event.json").write_text(
                json.dumps(event.to_dict(), indent=2), encoding="utf-8"
            )
        except OSError:
            logger.exception("cannot create snapshot folder %s", folder)
            return
        for i, (_ts, jpeg) in enumerate(list(self._ring)):
            self._write(folder, f"pre_{i:02d}.jpg", jpeg)
        if self._post_s > 0:
            with self._lock:
                self._pending.append(
                    _Pending(
                        event, folder, self._latest_ts + self._post_s + 1e-6, self._latest_ts, 0
                    )
                )

    def _prune(self) -> None:
        """Keep only the newest ``max_events - 1`` incident folders (disk-space bound)."""
        if not self.root.is_dir():
            return
        folders = sorted(
            (p for p in self.root.iterdir() if p.is_dir() and is_safe_event_id(p.name)),
            key=lambda p: p.stat().st_mtime,
        )
        for old in folders[: max(0, len(folders) - self._max_events + 1)]:
            try:
                shutil.rmtree(old)
            except OSError:
                logger.exception("cannot remove old snapshots %s", old)

    def list_snapshots(self, event_id: str) -> list[str]:
        if not is_safe_event_id(event_id):
            return []
        folder = self.root / event_id
        if not folder.is_dir():
            return []
        return sorted(p.name for p in folder.glob("*.jpg"))

    def snapshot_path(self, event_id: str, name: str) -> Path | None:
        if not is_safe_event_id(event_id) or name not in self.list_snapshots(event_id):
            return None
        return self.root / event_id / name

    @staticmethod
    def _write(folder: Path, name: str, data: bytes) -> None:
        try:
            (folder / name).write_bytes(data)
        except OSError:
            logger.exception("cannot write snapshot %s", folder / name)
