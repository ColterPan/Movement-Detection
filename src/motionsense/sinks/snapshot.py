"""Save before/after frames for incident events.

For every incident two versions of each frame are written:

* ``before_01.jpg`` / ``after_01.jpg`` are **annotated** (skeletons, ids, activity, fall risk)
  so a person can see exactly what the detector saw;
* ``before_01_raw.jpg`` / ``after_01_raw.jpg`` are the untouched camera frames (for a
  second opinion from a vision model in phase 2).

``frames.json`` lists every frame with its time offset relative to the event. Frames live in
a small in-memory ring buffer; nothing is written to disk unless an incident happens.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from motionsense.events.models import INCIDENT_TYPES, Event
from motionsense.render.overlay import draw_scene, encode_jpeg
from motionsense.types import Frame, SceneState

logger = logging.getLogger(__name__)

_SAFE_ID = re.compile(r"^[0-9a-f]{32}$")
_SAFE_NAME = re.compile(r"^(before|after)_\d{2}(_raw)?\.jpg$")
MANIFEST = "frames.json"


def is_safe_event_id(event_id: str) -> bool:
    return bool(_SAFE_ID.match(event_id))


@dataclass(frozen=True, slots=True)
class _Shot:
    ts: float
    annotated: bytes
    raw: bytes


@dataclass
class _Incident:
    event: Event
    folder: Path
    until: float
    next_at: float
    frames: list[dict[str, Any]] = field(default_factory=list)


class SnapshotRecorder:
    def __init__(
        self,
        root: Path,
        pre_s: float = 2.0,
        post_s: float = 3.0,
        hz: float = 3.0,
        jpeg_quality: int = 85,
        max_events: int = 100,
    ) -> None:
        self.root = root
        self._pre_s = pre_s
        self._post_s = post_s
        self._interval = 1.0 / hz
        self._max_events = max_events
        self._quality = jpeg_quality
        self._ring: deque[_Shot] = deque()
        self._last_sample = float("-inf")
        self._latest_ts = 0.0
        self._open: list[_Incident] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ frames (pipeline thread)
    def on_frame(self, frame: Frame, scene: SceneState) -> None:
        self._latest_ts = frame.ts
        due_ring = frame.ts - self._last_sample >= self._interval
        with self._lock:
            due = [i for i in self._open if frame.ts >= i.next_at]
        if not due_ring and not due:
            return
        shot = _Shot(
            ts=frame.ts,
            annotated=encode_jpeg(draw_scene(np.asarray(frame.image), scene), self._quality),
            raw=encode_jpeg(np.asarray(frame.image), self._quality),
        )
        if due_ring:
            self._last_sample = frame.ts
            self._ring.append(shot)
            while self._ring and self._ring[0].ts < frame.ts - self._pre_s:
                self._ring.popleft()
        for incident in due:
            self._save(incident, "after", shot)
            self._write_manifest(incident)
            incident.next_at = frame.ts + self._interval
        with self._lock:
            self._open = [i for i in self._open if frame.ts < i.until]

    # ------------------------------------------------------------------ events (bus)
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
        incident = _Incident(event, folder, self._latest_ts + self._post_s + 1e-6, self._latest_ts)
        for shot in list(self._ring):
            self._save(incident, "before", shot)
        self._write_manifest(incident)
        if self._post_s > 0:
            with self._lock:
                self._open.append(incident)

    def _save(self, incident: _Incident, phase: str, shot: _Shot) -> None:
        n = sum(1 for f in incident.frames if f["phase"] == phase) + 1
        name = f"{phase}_{n:02d}.jpg"
        raw_name = f"{phase}_{n:02d}_raw.jpg"
        self._write(incident.folder, name, shot.annotated)
        self._write(incident.folder, raw_name, shot.raw)
        incident.frames.append(
            {
                "name": name,
                "raw": raw_name,
                "phase": phase,
                "offset_s": round(shot.ts - incident.event.ts, 2),
            }
        )

    def _write_manifest(self, incident: _Incident) -> None:
        self._write(
            incident.folder, MANIFEST, json.dumps(incident.frames, indent=2).encode("utf-8")
        )

    # ------------------------------------------------------------------ housekeeping
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

    # ------------------------------------------------------------------ queries (API)
    def frames(self, event_id: str) -> list[dict[str, Any]]:
        """Frames of an incident in time order: name, raw, phase, offset_s."""
        if not is_safe_event_id(event_id):
            return []
        folder = self.root / event_id
        manifest = folder / MANIFEST
        if not manifest.is_file():
            return []
        try:
            frames = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.exception("unreadable snapshot manifest %s", manifest)
            return []
        return [f for f in frames if (folder / f["name"]).is_file()]

    def event_info(self, event_id: str) -> dict[str, Any] | None:
        """The saved event (available even after a restart, unlike the in-memory bus)."""
        if not is_safe_event_id(event_id):
            return None
        path = self.root / event_id / "event.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def incidents(self, limit: int = 20) -> list[dict[str, Any]]:
        """Saved incidents, newest first (survives restarts)."""
        if not self.root.is_dir():
            return []
        folders = sorted(
            (p for p in self.root.iterdir() if p.is_dir() and is_safe_event_id(p.name)),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        out: list[dict[str, Any]] = []
        for folder in folders[:limit]:
            info = self.event_info(folder.name)
            if info is not None:
                out.append(info)
        return out

    def list_snapshots(self, event_id: str) -> list[str]:
        """Annotated frame names in time order."""
        return [f["name"] for f in self.frames(event_id)]

    def snapshot_path(self, event_id: str, name: str) -> Path | None:
        if not is_safe_event_id(event_id) or not _SAFE_NAME.match(name):
            return None
        path = self.root / event_id / name
        return path if path.is_file() else None

    @staticmethod
    def _write(folder: Path, name: str, data: bytes) -> None:
        try:
            (folder / name).write_bytes(data)
        except OSError:
            logger.exception("cannot write snapshot %s", folder / name)
