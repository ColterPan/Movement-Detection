"""Activity classifier interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from motionsense.features.geometry import PoseFeatures
from motionsense.features.kinematics import MotionFeatures

ACTIVITIES: tuple[str, ...] = (
    "standing",
    "sitting",
    "lying",
    "walking",
    "running",
    "squatting",
    "push_up",
)


@dataclass(frozen=True, slots=True)
class ActivityContext:
    """Scene context that can disambiguate poses (e.g. sitting on a chair vs squatting)."""

    support: str | None = None  # furniture label under the hips, if any


@dataclass(frozen=True, slots=True)
class ActivityResult:
    probabilities: dict[str, float]  # sums to 1 over ACTIVITIES
    quality: str  # "full" | "partial" | "low"
    attributes: dict[str, object] = field(default_factory=dict)

    @property
    def top(self) -> tuple[str, float]:
        name = max(self.probabilities, key=lambda k: self.probabilities[k])
        return name, self.probabilities[name]


class ActivityClassifier(Protocol):
    """Maps pose + motion features to a probability per activity (stateless per call)."""

    def classify(
        self, pose: PoseFeatures, motion: MotionFeatures, context: ActivityContext
    ) -> ActivityResult: ...
