"""Scripted motion scenarios: a timeline of segments mapping time -> PoseParams.

Each scenario also declares the furniture visible in the scene and the expected
ground truth (whether it contains a fall and when), so it doubles as a labelled test clip.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from motionsense.simulation.skeleton import LIE, PLANK, SIT, SQUAT, STAND, PoseParams
from motionsense.types import ObjectDetection

SegmentFn = Callable[[float], PoseParams]  # local time (s) -> pose


@dataclass(frozen=True, slots=True)
class Segment:
    duration: float
    fn: SegmentFn
    label: str


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    segments: tuple[Segment, ...]
    objects: tuple[ObjectDetection, ...] = ()
    has_fall: bool = False
    fall_time: float | None = None
    description: str = ""
    expected_labels: tuple[str, ...] = field(default_factory=tuple)

    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.segments)

    def segment_at(self, t: float) -> tuple[Segment, float]:
        t = min(max(t, 0.0), self.duration - 1e-9)
        for seg in self.segments:
            if t < seg.duration:
                return seg, t
            t -= seg.duration
        last = self.segments[-1]
        return last, last.duration

    def pose_at(self, t: float) -> PoseParams:
        seg, local = self.segment_at(t)
        return seg.fn(local)

    def label_at(self, t: float) -> str:
        return self.segment_at(t)[0].label


# ---------------------------------------------------------------- segment builders
def _smooth(a: float) -> float:
    a = min(max(a, 0.0), 1.0)
    return a * a * (3 - 2 * a)


def hold(pose: PoseParams, duration: float, label: str) -> Segment:
    return Segment(duration, lambda _t: pose, label)


def move(a: PoseParams, b: PoseParams, duration: float, label: str, smooth: bool = True) -> Segment:
    def fn(t: float) -> PoseParams:
        x = t / duration
        return a.lerp(b, _smooth(x) if smooth else x)

    return Segment(duration, fn, label)


def walk(x0: float, x1: float, duration: float, *, run: bool = False) -> Segment:
    freq = 1.4 if run else 0.9  # per-leg swing frequency (Hz)
    amp = 40.0 if run else 22.0
    tilt = 10.0 if run else 2.0

    def fn(t: float) -> PoseParams:
        phase = math.sin(2 * math.pi * freq * t)
        bend = abs(phase)
        return STAND.with_(
            x=x0 + (x1 - x0) * t / duration,
            tilt=tilt,
            leg_swing=amp * phase,
            knee=(150.0 - 45.0 * bend) if run else (175.0 - 18.0 * bend),
            shoulder=15.0 + (25.0 if run else 12.0) * phase,
            elbow=95.0 if run else 160.0,
        )

    return Segment(duration, fn, "running" if run else "walking")


def oscillate(a: PoseParams, b: PoseParams, period: float, reps: int, label: str) -> Segment:
    def fn(t: float) -> PoseParams:
        x = 0.5 - 0.5 * math.cos(2 * math.pi * t / period)
        return a.lerp(b, x)

    return Segment(period * reps, fn, label)


def look_around(duration: float) -> Segment:
    return Segment(
        duration, lambda t: STAND.with_(yaw=0.9 * math.sin(2 * math.pi * 0.5 * t)), "standing"
    )


HIDDEN = STAND.with_(visible=0.0)

# ---------------------------------------------------------------- furniture
FRAME_W, FRAME_H = 1280, 720
BED = ObjectDetection("bed", 0.9, (420.0, 470.0, 1100.0, 640.0))
CHAIR = ObjectDetection("chair", 0.85, (560.0, 430.0, 760.0, 670.0))


def _bed_lie(x: float) -> PoseParams:
    return LIE.with_(x=x, lift=1.25)


# ---------------------------------------------------------------- scenarios
def _scenarios() -> dict[str, Scenario]:
    stand_c = STAND.with_(x=0.5)
    fallen = LIE.with_(x=0.62, hip=165.0, knee=150.0)
    stand_l = STAND.with_(x=0.22)
    plank_l = PLANK.with_(x=0.22)
    fallen_l = fallen.with_(x=0.3)
    return {
        "standing": Scenario(
            "standing", (hold(stand_c, 8.0, "standing"),), expected_labels=("standing",)
        ),
        "walk": Scenario(
            "walk", (walk(0.15, 0.85, 5.0), walk(0.85, 0.15, 5.0)), expected_labels=("walking",)
        ),
        "run": Scenario(
            "run",
            (walk(0.1, 0.9, 2.5, run=True), walk(0.9, 0.1, 2.5, run=True)),
            expected_labels=("running",),
        ),
        "sit_down": Scenario(
            "sit_down",
            (
                hold(stand_c, 2.0, "standing"),
                move(stand_c, SIT, 1.2, "sitting"),
                hold(SIT, 5.0, "sitting"),
            ),
            objects=(CHAIR,),
            expected_labels=("sitting",),
        ),
        "squats": Scenario(
            "squats",
            (hold(stand_c, 1.5, "standing"), oscillate(stand_c, SQUAT, 2.0, 5, "squatting")),
            expected_labels=("squatting",),
        ),
        "push_ups": Scenario(
            "push_ups",
            (
                hold(PLANK, 1.0, "push_up"),
                oscillate(PLANK, PLANK.with_(tilt=84.0, elbow=80.0), 1.6, 6, "push_up"),
            ),
            expected_labels=("push_up",),
        ),
        "looking_around": Scenario(
            "looking_around", (look_around(8.0),), expected_labels=("standing",)
        ),
        "fall": Scenario(
            "fall",
            (
                hold(stand_c, 2.0, "standing"),
                walk(0.5, 0.58, 1.0),
                move(STAND.with_(x=0.58), fallen, 0.55, "falling", smooth=False),
                hold(fallen, 10.0, "lying"),
            ),
            has_fall=True,
            fall_time=3.0,
            expected_labels=("lying",),
        ),
        "fall_get_up": Scenario(
            "fall_get_up",
            (
                hold(stand_c, 2.0, "standing"),
                move(stand_c, fallen, 0.5, "falling", smooth=False),
                hold(fallen, 5.0, "lying"),
                move(fallen, SIT.with_(x=0.6), 1.5, "sitting"),
                move(SIT.with_(x=0.6), STAND.with_(x=0.6), 1.2, "standing"),
                hold(STAND.with_(x=0.6), 4.0, "standing"),
            ),
            has_fall=True,
            fall_time=2.0,
        ),
        "fall_out_of_view": Scenario(
            "fall_out_of_view",
            (
                hold(stand_c, 2.0, "standing"),
                move(stand_c, fallen, 0.35, "falling", smooth=False),
                hold(HIDDEN, 5.0, "hidden"),
            ),
            has_fall=True,
            fall_time=2.0,
        ),
        "stumble": Scenario(
            "stumble",
            (
                hold(stand_c, 2.0, "standing"),
                move(stand_c, SQUAT.with_(tilt=45.0), 0.4, "squatting", smooth=False),
                move(SQUAT.with_(tilt=45.0), stand_c, 0.7, "standing"),
                hold(stand_c, 5.0, "standing"),
            ),
        ),
        "lie_on_bed": Scenario(
            "lie_on_bed",
            (
                hold(STAND.with_(x=0.5), 2.0, "standing"),
                move(STAND.with_(x=0.5), _bed_lie(0.62), 0.6, "lying", smooth=False),
                hold(_bed_lie(0.62), 10.0, "lying"),
            ),
            objects=(BED,),
            expected_labels=("lying",),
        ),
        "lie_down_slowly": Scenario(
            "lie_down_slowly",
            (
                hold(stand_c, 2.0, "standing"),
                move(stand_c, SIT.with_(lift=0.0), 2.0, "sitting"),
                move(SIT, LIE.with_(x=0.6), 4.0, "lying"),
                hold(LIE.with_(x=0.6), 40.0, "lying"),
            ),
            expected_labels=("lying",),
        ),
        "demo": Scenario(
            "demo",
            (
                hold(stand_c, 3.0, "standing"),
                look_around(4.0),
                walk(0.5, 0.85, 3.0),
                walk(0.85, 0.15, 2.5, run=True),
                walk(0.15, 0.5, 3.0),
                move(stand_c, SIT, 1.2, "sitting"),
                hold(SIT, 4.0, "sitting"),
                move(SIT, stand_c, 1.0, "standing"),
                walk(0.5, 0.22, 2.5),
                # away from the chair, so squats are not mistaken for sitting down
                oscillate(stand_l, SQUAT.with_(x=0.22), 2.0, 3, "squatting"),
                move(stand_l, plank_l, 1.5, "push_up"),
                oscillate(plank_l, plank_l.with_(tilt=84.0, elbow=80.0), 1.6, 4, "push_up"),
                move(plank_l, stand_l, 1.5, "standing"),
                hold(stand_l, 2.0, "standing"),
                move(stand_l, fallen_l, 0.5, "falling", smooth=False),
                hold(fallen_l, 6.0, "lying"),
                move(fallen_l, stand_l, 2.0, "standing"),
                walk(0.22, 0.5, 2.5),
                hold(stand_c, 2.0, "standing"),
            ),
            objects=(CHAIR,),
        ),
    }


SCENARIOS: dict[str, Scenario] = _scenarios()


def get_scenario(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError as exc:
        raise ValueError(f"unknown scenario {name!r}; choose from {sorted(SCENARIOS)}") from exc
