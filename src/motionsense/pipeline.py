"""The analysis pipeline: perception -> tracking -> features -> activity + fall -> events.

``Pipeline.process(frame)`` is synchronous and deterministic given the backends, which makes
it easy to test and to run offline on video files. ``motionsense.service`` runs it live.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

from motionsense.activity.base import ActivityClassifier, ActivityContext
from motionsense.activity.rep_counter import PUSH_UP, SQUAT, RepCounter
from motionsense.activity.rule_classifier import RuleBasedClassifier
from motionsense.activity.smoothing import ProbabilitySmoother, StableLabel
from motionsense.config import Settings
from motionsense.events.bus import EventBus
from motionsense.events.models import INCIDENT_TYPES, Event, EventType, make_event
from motionsense.fall.context import LYING_SUPPORTS, find_support
from motionsense.fall.detector import FallDetector, FallSignal, FallState
from motionsense.features.geometry import compute_pose_features
from motionsense.features.kinematics import compute_motion_features
from motionsense.perception.base import ObjectBackend, PoseBackend
from motionsense.tracking.history import Sample, TrackHistory
from motionsense.tracking.ids import IouTracker, iou
from motionsense.types import (
    EMPTY_SCENE,
    BBox,
    Frame,
    ObjectDetection,
    PersonDetection,
    PersonState,
    SceneState,
)

logger = logging.getLogger(__name__)

FrameListener = Callable[[Frame, SceneState], None]
ANNOUNCE_AFTER_S = 0.5
FURNITURE_MEMORY_S = 60.0
FPS_EMA = 0.1
DOWN_MEMORY_S = 60.0
DOWN_MATCH_IOU = 0.2
DOWN_STATES = frozenset({FallState.FALLEN, FallState.INACTIVE})
GHOST_HOLD_S = 0.6  # keep showing a person this long after a missed detection
CLOSEUP_AREA = 0.25  # a person box covering this much of the frame is a close-up
INSIDE_PERSON = 0.6  # objects mostly inside a close-up person are ignored


def _area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersection(a: BBox, b: BBox) -> float:
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


@dataclass
class TrackContext:
    history: TrackHistory
    smoother: ProbabilitySmoother
    label: StableLabel
    reps: dict[str, RepCounter]
    fall: FallDetector
    first_seen: float
    last_seen: float
    announced: bool = False
    last_state: PersonState | None = field(default=None, repr=False)


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        pose_backend: PoseBackend,
        object_backend: ObjectBackend | None = None,
        bus: EventBus | None = None,
        classifier: ActivityClassifier | None = None,
        listeners: Sequence[FrameListener] = (),
    ) -> None:
        self.settings = settings
        self.bus = bus or EventBus()
        self._pose = pose_backend
        self._objects_backend = object_backend if settings.objects.enabled else None
        self._classifier = classifier or RuleBasedClassifier(
            settings.activity.looking_around_yaw_std
        )
        self._listeners = list(listeners)
        self._tracks: dict[int, TrackContext] = {}
        self._down_tracks: list[tuple[FallDetector, BBox, float]] = []
        self._id_fallback = IouTracker()
        self._objects: tuple[ObjectDetection, ...] = ()
        self._furniture: list[tuple[ObjectDetection, float]] = []
        self._last_object_ts = float("-inf")
        self._fps = 0.0
        self._last_wall: float | None = None
        self.latest: SceneState = EMPTY_SCENE

    # ------------------------------------------------------------------ public
    def add_listener(self, listener: FrameListener) -> None:
        self._listeners.append(listener)

    def process(self, frame: Frame) -> SceneState:
        self._update_fps()
        self._maybe_detect_objects(frame)
        detections = self._id_fallback.assign(self._pose.infer(frame))

        persons: list[PersonState] = []
        seen: set[int] = set()
        for det in detections:
            if det.track_id is None:  # pragma: no cover - IouTracker always assigns
                continue
            seen.add(det.track_id)
            persons.append(self._analyse_person(det, frame))

        persons.extend(self._recently_missed(frame.ts, seen))
        self._expire_tracks(frame.ts, seen)

        state = SceneState(
            ts=frame.ts,
            frame_index=frame.index,
            frame_size=frame.size,
            fps=self._fps,
            persons=tuple(persons),
            objects=self._objects,
        )
        self.latest = state
        for listener in self._listeners:
            try:
                listener(frame, state)
            except Exception:
                logger.exception("frame listener %r failed", listener)
        return state

    def flush(self, ts: float) -> None:
        """Expire every track (end of a video file)."""
        self._expire_tracks(ts + 1e6, set())

    # ------------------------------------------------------------------ internals
    def _update_fps(self) -> None:
        now = time.perf_counter()
        if self._last_wall is not None:
            dt = now - self._last_wall
            if dt > 0:
                inst = 1.0 / dt
                self._fps = inst if self._fps == 0 else (1 - FPS_EMA) * self._fps + FPS_EMA * inst
        self._last_wall = now

    def _maybe_detect_objects(self, frame: Frame) -> None:
        if self._objects_backend is None:
            return
        if frame.ts - self._last_object_ts < self.settings.objects.interval_s:
            return
        self._last_object_ts = frame.ts
        try:
            fresh = self._without_closeup_ghosts(self._objects_backend.infer(frame), frame.size)
        except Exception:
            logger.exception("object detection failed")
            return
        self._objects = tuple(fresh) + tuple(self._remembered_furniture(fresh, frame.ts))

    def _without_closeup_ghosts(
        self, objects: list[ObjectDetection], frame_size: tuple[int, int]
    ) -> list[ObjectDetection]:
        """Drop "objects" that are really parts of a person filling the view (a close-up torso
        is easily mistaken for a chair, bed or toilet)."""
        frame_area = max(frame_size[0] * frame_size[1], 1)
        big_people = [
            p.bbox for p in self.latest.persons if _area(p.bbox) / frame_area >= CLOSEUP_AREA
        ]
        if not big_people:
            return objects
        return [
            o
            for o in objects
            if all(
                _intersection(o.bbox, b) / max(_area(o.bbox), 1.0) < INSIDE_PERSON
                for b in big_people
            )
        ]

    def _recently_missed(self, ts: float, seen: set[int]) -> list[PersonState]:
        """People the detector missed for a moment stay on screen briefly (no flicker)."""
        out: list[PersonState] = []
        for tid, ctx in self._tracks.items():
            if tid in seen or ctx.last_state is None or ts - ctx.last_seen > GHOST_HOLD_S:
                continue
            out.append(
                replace(ctx.last_state, attributes={**ctx.last_state.attributes, "stale": True})
            )
        return out

    def _remembered_furniture(
        self, fresh: list[ObjectDetection], ts: float
    ) -> list[ObjectDetection]:
        """Keep recently seen beds/couches that are now hidden (e.g. a person lies on the bed).
        Small furniture (chairs) is not remembered: it moves, and false detections would linger."""
        support = LYING_SUPPORTS & set(self.settings.objects.support_classes)
        kept: list[tuple[ObjectDetection, float]] = [(o, ts) for o in fresh if o.label in support]
        remembered: list[ObjectDetection] = []
        for old, seen_ts in self._furniture:
            if ts - seen_ts > FURNITURE_MEMORY_S:
                continue
            matched = any(o.label == old.label and iou(o.bbox, old.bbox) > 0.3 for o in fresh)
            if not matched:
                kept.append((old, seen_ts))
                remembered.append(old)
        self._furniture = kept
        return remembered

    def _inherit_down_state(self, bbox: BBox, ts: float) -> FallDetector | None:
        self._down_tracks = [d for d in self._down_tracks if ts - d[2] <= DOWN_MEMORY_S]
        for i, (detector, old_box, _lost) in enumerate(self._down_tracks):
            if iou(bbox, old_box) >= DOWN_MATCH_IOU:
                del self._down_tracks[i]
                return detector
        return None

    def _new_track(self, ts: float) -> TrackContext:
        cfg = self.settings
        return TrackContext(
            history=TrackHistory(),
            smoother=ProbabilitySmoother(cfg.activity.ema_tau_s),
            label=StableLabel(cfg.activity.change_hold_s),
            reps={"push_up": RepCounter(PUSH_UP), "squat": RepCounter(SQUAT)},
            fall=FallDetector(cfg.fall),
            first_seen=ts,
            last_seen=ts,
        )

    def _analyse_person(self, det: PersonDetection, frame: Frame) -> PersonState:
        if det.track_id is None:  # pragma: no cover - IouTracker always assigns an id
            raise ValueError("detection without a track id")
        ts = frame.ts
        tid = det.track_id
        ctx = self._tracks.get(tid)
        if ctx is None:
            ctx = self._new_track(ts)
            inherited = self._inherit_down_state(det.bbox, ts)
            if inherited is not None:
                ctx.fall = inherited
            self._tracks[tid] = ctx
        ctx.last_seen = ts

        pose = compute_pose_features(
            det.keypoints, det.bbox, self.settings.models.keypoint_min_conf, frame_size=frame.size
        )
        ctx.history.add(Sample(ts=ts, pose=pose, keypoints=det.keypoints, bbox=det.bbox))
        motion = compute_motion_features(ctx.history)
        support = find_support(pose.hip_xy, self._objects, self.settings.objects.support_classes)

        result = self._classifier.classify(pose, motion, ActivityContext(support=support))
        probs = ctx.smoother.update(result.probabilities, ts)
        top = max(probs, key=lambda k: probs[k])

        if not ctx.announced and ts - ctx.first_seen >= ANNOUNCE_AFTER_S:
            ctx.announced = True
            self._emit(EventType.PERSON_ENTERED, ts, tid, None, f"Person #{tid} entered the view")

        changed = ctx.label.update(top, ts)
        if changed is not None and ctx.announced:
            self._emit(
                EventType.ACTIVITY_CHANGED,
                ts,
                tid,
                probs[changed],
                f"Person #{tid} is now {changed.replace('_', ' ')}",
                {"activity": changed},
            )

        self._count_reps(ctx, tid, ts, pose.elbow_angle, pose.knee_angle, probs)

        fall = ctx.fall.update(ts, pose, motion, probs, support)
        for signal in fall.signals:
            self._emit_fall(signal, ts, tid)

        state = PersonState(
            track_id=tid,
            bbox=det.bbox,
            keypoints=det.keypoints,
            activities=probs,
            top_activity=top,
            top_confidence=probs[top],
            quality=result.quality,
            fall_state=fall.state.value,
            fall_risk=fall.risk,
            reps={name: c.count for name, c in ctx.reps.items()},
            attributes=dict(result.attributes),
            support=support,
            features={
                "torso_tilt": pose.torso_tilt,
                "speed": motion.body_speed,
                "drop_1s": motion.drop_1s,
                "impact": fall.impact_score,
            },
        )
        ctx.last_state = state
        return state

    def _count_reps(
        self,
        ctx: TrackContext,
        tid: int,
        ts: float,
        elbow: float,
        knee: float,
        probs: dict[str, float],
    ) -> None:
        for name, angle in (("push_up", elbow), ("squat", knee)):
            counter = ctx.reps[name]
            if counter.update(angle, probs.get(counter.spec.activity, 0.0)):
                self._emit(
                    EventType.REP_COMPLETED,
                    ts,
                    tid,
                    None,
                    f"Person #{tid}: {name.replace('_', ' ')} rep {counter.count}",
                    {"exercise": name, "count": counter.count},
                )

    def _expire_tracks(self, ts: float, seen: set[int]) -> None:
        lost_after = self.settings.fall.track_lost_s
        for tid in [
            t for t, c in self._tracks.items() if t not in seen and ts - c.last_seen > lost_after
        ]:
            ctx = self._tracks.pop(tid)
            update = ctx.fall.on_track_lost(ts)
            for signal in update.signals:
                self._emit_fall(signal, ctx.last_seen, tid)
            latest = ctx.history.latest
            if ctx.fall.state in DOWN_STATES and latest is not None:
                # Someone on the floor often flickers out of detection; if they reappear
                # nearby under a new id, the new track continues their fall incident.
                self._down_tracks.append((ctx.fall, latest.bbox, ts))
            if ctx.announced:
                self._emit(EventType.PERSON_LEFT, ts, tid, None, f"Person #{tid} left the view")

    def _emit_fall(self, signal: FallSignal, ts: float, tid: int) -> None:
        pct = f"{signal.confidence * 100:.0f}%"
        messages = {
            EventType.FALL_SUSPECTED: f"Possible fall: person #{tid} dropped suddenly ({pct})",
            EventType.FALL_CONFIRMED: f"FALL DETECTED: person #{tid} is down ({pct})",
            EventType.FALL_RECOVERED: f"Person #{tid} got back up",
            EventType.INACTIVE_ON_FLOOR: f"ALERT: person #{tid} has been on the floor and not getting up",
        }
        self._emit(
            signal.type,
            ts,
            tid,
            signal.confidence,
            messages.get(signal.type, signal.type.value),
            signal.details,
        )

    def _emit(
        self,
        type_: EventType,
        ts: float,
        tid: int | None,
        confidence: float | None,
        message: str,
        details: dict | None = None,
    ) -> Event:
        event = make_event(
            type_,
            ts=ts,
            track_id=tid,
            confidence=confidence,
            message=message,
            details=details,
            camera_id=self.settings.camera.camera_id,
        )
        if self.settings.sinks.snapshots and event.type in INCIDENT_TYPES:
            event = event.model_copy(
                update={"snapshots_url": f"/api/v1/events/{event.id}/snapshots"}
            )
        return self.bus.publish(event)

    @property
    def active_fall_states(self) -> dict[int, FallState]:
        return {tid: ctx.fall.state for tid, ctx in self._tracks.items()}
