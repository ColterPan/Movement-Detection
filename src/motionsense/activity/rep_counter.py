"""Exercise repetition counting with angle hysteresis."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RepSpec:
    name: str
    activity: str  # activity that must be likely for reps to count
    min_activity_prob: float
    down_below: float  # joint angle (deg) that marks the bottom of a rep
    up_above: float  # joint angle that marks the top of a rep


PUSH_UP = RepSpec("push_up", "push_up", 0.30, down_below=100.0, up_above=145.0)
SQUAT = RepSpec("squat", "squatting", 0.20, down_below=110.0, up_above=150.0)


class RepCounter:
    """Counts down→up cycles of a joint angle while the matching activity is likely."""

    def __init__(self, spec: RepSpec) -> None:
        self.spec = spec
        self.count = 0
        self._phase = "up"

    def update(self, angle: float, activity_prob: float) -> bool:
        """Feed one observation. Returns True when a repetition completes."""
        if math.isnan(angle):
            return False
        if self._phase == "up":
            if angle < self.spec.down_below and activity_prob >= self.spec.min_activity_prob:
                self._phase = "down"
            return False
        if angle > self.spec.up_above:
            self._phase = "up"
            if activity_prob >= self.spec.min_activity_prob * 0.5:
                self.count += 1
                return True
        return False
