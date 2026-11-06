"""Fallback IoU tracker for pose backends that do not provide track ids."""

from __future__ import annotations

from dataclasses import replace

from motionsense.types import BBox, PersonDetection

MIN_IOU = 0.3


def iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class IouTracker:
    """Gives id-less detections a stable id by IoU matching against recently seen boxes.

    Backends such as ByteTrack occasionally emit a frame without ids (e.g. low-confidence
    detections of someone lying on the floor). Matching against *all* recently seen tracks,
    including the backend's own ids, keeps the person's identity, and so their fall state,
    stable across those frames.
    """

    def __init__(self, start_id: int = 1_000_000, memory_frames: int = 30) -> None:
        self._next_id = start_id
        self._memory = memory_frames
        self._frame = 0
        self._recent: dict[int, tuple[BBox, int]] = {}  # track id -> (last box, frame seen)

    def assign(self, detections: list[PersonDetection]) -> list[PersonDetection]:
        self._frame += 1
        out = detections
        if any(d.track_id is None for d in detections):
            out = self._match(detections)
        for d in out:
            if d.track_id is not None:
                self._recent[d.track_id] = (d.bbox, self._frame)
        self._recent = {
            tid: v for tid, v in self._recent.items() if self._frame - v[1] <= self._memory
        }
        return out

    def _match(self, detections: list[PersonDetection]) -> list[PersonDetection]:
        taken = {d.track_id for d in detections if d.track_id is not None}
        candidates = sorted(
            (
                (iou(d.bbox, box), i, tid)
                for i, d in enumerate(detections)
                if d.track_id is None
                for tid, (box, _seen) in self._recent.items()
                if tid not in taken
            ),
            reverse=True,
        )
        assigned: dict[int, int] = {}
        used: set[int] = set()
        for score, i, tid in candidates:
            if score < MIN_IOU or i in assigned or tid in used:
                continue
            assigned[i] = tid
            used.add(tid)
        out: list[PersonDetection] = []
        for i, d in enumerate(detections):
            if d.track_id is not None:
                out.append(d)
                continue
            new_id = assigned.get(i)
            if new_id is None:
                new_id = self._next_id
                self._next_id += 1
            out.append(replace(d, track_id=new_id))
        return out
