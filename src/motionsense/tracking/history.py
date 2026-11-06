"""Per-track ring buffer of recent observations."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass

from motionsense.features.geometry import PoseFeatures
from motionsense.types import BBox, Keypoints

DEFAULT_HISTORY_SECONDS = 6.0


@dataclass(frozen=True, slots=True)
class Sample:
    ts: float
    pose: PoseFeatures
    keypoints: Keypoints
    bbox: BBox

    @property
    def raw_scale(self) -> float:
        """Torso length if known, else a third of the box height (rough body proportion)."""
        if not math.isnan(self.pose.torso_len) and self.pose.torso_len > 1.0:
            return self.pose.torso_len
        return max((self.bbox[3] - self.bbox[1]) / 3.0, 1.0)


class TrackHistory:
    """Time-ordered samples for a single track, trimmed to ``max_seconds``."""

    def __init__(self, max_seconds: float = DEFAULT_HISTORY_SECONDS) -> None:
        self._max_seconds = max_seconds
        self._samples: deque[Sample] = deque()

    def add(self, sample: Sample) -> None:
        if self._samples and sample.ts <= self._samples[-1].ts:
            # Out-of-order or duplicate timestamp: replace the newest sample.
            self._samples.pop()
        self._samples.append(sample)
        cutoff = sample.ts - self._max_seconds
        while self._samples and self._samples[0].ts < cutoff:
            self._samples.popleft()

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples)

    @property
    def latest(self) -> Sample | None:
        return self._samples[-1] if self._samples else None

    @property
    def last_ts(self) -> float:
        return self._samples[-1].ts if self._samples else -math.inf

    def window(self, seconds: float) -> list[Sample]:
        """Samples within ``seconds`` of the newest one (inclusive), oldest first."""
        if not self._samples:
            return []
        cutoff = self._samples[-1].ts - seconds
        return [s for s in self._samples if s.ts >= cutoff]
