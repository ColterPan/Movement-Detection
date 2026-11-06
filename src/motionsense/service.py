"""Composition root: builds sources, backends, pipeline and sinks from Settings and runs the
pipeline on a background thread. The API layer only talks to ``MotionService``."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from motionsense.config import Settings
from motionsense.events.bus import EventBus
from motionsense.events.models import EventType, Severity, make_event
from motionsense.perception.base import ObjectBackend, PoseBackend
from motionsense.pipeline import Pipeline
from motionsense.render.overlay import draw_scene, encode_jpeg
from motionsense.sinks.jsonl_log import JsonlEventLog
from motionsense.sinks.snapshot import SnapshotRecorder
from motionsense.sinks.webhook import WebhookSink
from motionsense.sources.base import FrameSource
from motionsense.sources.opencv_source import redact_url
from motionsense.types import EMPTY_SCENE, Frame, SceneState

logger = logging.getLogger(__name__)

SYNTHETIC_PREFIX = "synthetic"
MAX_CONSECUTIVE_FAILURES = 30


@dataclass(frozen=True, slots=True)
class Components:
    source: FrameSource
    pose: PoseBackend
    objects: ObjectBackend | None


def _is_synthetic(source: str) -> bool:
    return source == SYNTHETIC_PREFIX or source.startswith(SYNTHETIC_PREFIX + ":")


def build_components(settings: Settings) -> Components:
    """Create the frame source and perception backends described by ``settings``."""
    spec = settings.camera.source.strip()
    if _is_synthetic(spec):
        from motionsense.simulation.backend import (
            ScriptedObjectBackend,
            ScriptedPoseBackend,
            SyntheticSource,
        )
        from motionsense.simulation.scenarios import get_scenario

        scenario = get_scenario(spec.split(":", 1)[1] if ":" in spec else "demo")
        return Components(
            source=SyntheticSource(scenario, fps=15.0, realtime=True, loop=True),
            pose=ScriptedPoseBackend(scenario, noise_px=1.5, loop=True),
            objects=ScriptedObjectBackend(scenario),
        )

    from motionsense.perception.yolo_objects import YoloObjectBackend
    from motionsense.perception.yolo_pose import YoloPoseBackend
    from motionsense.sources.opencv_source import CameraSource, VideoFileSource

    cam = settings.camera
    source: FrameSource
    if spec.isdigit():
        source = CameraSource(int(spec), cam.width, cam.height, cam.fps)
    elif spec.startswith(("rtsp://", "http://", "https://", "rtmp://")):
        source = CameraSource(spec, cam.width, cam.height, cam.fps)
    else:
        source = VideoFileSource(Path(spec), realtime=cam.realtime_files)

    pose = YoloPoseBackend(settings.models, settings.models_dir)
    objects = (
        YoloObjectBackend(settings.models, settings.models_dir)
        if settings.objects.enabled
        else None
    )
    return Components(source=source, pose=pose, objects=objects)


class MotionService:
    def __init__(self, settings: Settings, components: Components | None = None) -> None:
        self.settings = settings
        self.bus = EventBus()
        self._components = components
        self._pipeline: Pipeline | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._jpeg_lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._last_render = float("-inf")
        self.snapshots: SnapshotRecorder | None = None
        self._webhook: WebhookSink | None = None
        self.started_at: float | None = None
        self.error: str | None = None
        self.finished = False
        self._setup_sinks()

    # ------------------------------------------------------------------ setup
    def _setup_sinks(self) -> None:
        sinks = self.settings.sinks
        if sinks.jsonl:
            self.bus.subscribe(JsonlEventLog(self.settings.data_dir / "events.jsonl"))
        if sinks.snapshots:
            self.snapshots = SnapshotRecorder(
                self.settings.events_dir,
                pre_s=sinks.snapshot_pre_s,
                post_s=sinks.snapshot_post_s,
                hz=sinks.snapshot_hz,
                max_events=sinks.snapshot_max_events,
            )
            self.bus.subscribe(self.snapshots)
        if sinks.webhooks:
            self._webhook = WebhookSink(sinks.webhooks, Severity(sinks.webhook_min_severity))
            self.bus.subscribe(self._webhook)

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="motionsense-pipeline", daemon=True)
        self._thread.start()
        self.started_at = time.time()

    def stop(self) -> None:
        self._stop.set()
        if self._components is not None:
            self._components.source.close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if not self._thread.is_alive():
                self._thread = None
            else:  # still loading models; _run closes the source itself when it notices _stop
                logger.warning("pipeline thread did not stop within 5 s")
        if self._webhook is not None:
            self._webhook.close()

    def _system_error(self, message: str) -> None:
        """Record and broadcast that monitoring is impaired (dashboard, webhooks, log)."""
        self.error = message
        self.bus.publish(
            make_event(
                EventType.SYSTEM_ERROR,
                ts=time.time(),
                message=f"Monitoring impaired: {message}",
                camera_id=self.settings.camera.camera_id,
            )
        )

    def _run(self) -> None:
        comps: Components | None = None
        try:
            if self._components is None:
                logger.info("loading models and opening source ...")
                self._components = build_components(self.settings)
            comps = self._components
            if self._stop.is_set():
                return
            listeners = [self._render_listener]
            if self.snapshots is not None:
                listeners.append(self.snapshots.on_frame)
            self._pipeline = Pipeline(
                self.settings, comps.pose, comps.objects, bus=self.bus, listeners=listeners
            )
            logger.info("pipeline running on %s", comps.source.description)
            self._process_frames(comps, self._pipeline)
        except Exception as exc:
            logger.exception("pipeline crashed")
            self._system_error(f"{type(exc).__name__}: {exc}")
        finally:
            if comps is not None:
                comps.source.close()

    def _process_frames(self, comps: Components, pipeline: Pipeline) -> None:
        """Process frames; a single bad frame or transient model error must not stop monitoring."""
        last_ts = 0.0
        failures = 0
        for frame in comps.source.frames():
            if self._stop.is_set():
                break
            try:
                pipeline.process(frame)
            except Exception as exc:
                failures += 1
                logger.exception("frame %d failed (%d in a row)", frame.index, failures)
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    self._system_error(f"{failures} consecutive frame failures: {exc}")
                    return
                continue
            failures = 0
            last_ts = frame.ts
            source_error = getattr(comps.source, "error", None)
            if source_error and source_error != self.error:
                self._system_error(source_error)
            elif not source_error and self.error:
                self.error = None
        pipeline.flush(last_ts)
        self.finished = True

    def _render_listener(self, frame: Frame, scene: SceneState) -> None:
        now = time.perf_counter()
        if now - self._last_render < 1.0 / self.settings.api.stream_fps:
            return
        self._last_render = now
        jpeg = encode_jpeg(draw_scene(frame.image, scene), self.settings.api.jpeg_quality)
        with self._jpeg_lock:
            self._latest_jpeg = jpeg

    # ------------------------------------------------------------------ queries
    @property
    def latest_state(self) -> SceneState:
        return self._pipeline.latest if self._pipeline is not None else EMPTY_SCENE

    def latest_jpeg(self) -> bytes | None:
        with self._jpeg_lock:
            return self._latest_jpeg

    def status(self) -> dict[str, Any]:
        state = self.latest_state
        running = self._thread is not None and self._thread.is_alive()
        return {
            "status": "ok"
            if running and not self.error
            else ("finished" if self.finished else "degraded"),
            "running": running,
            "source": self._components.source.description
            if self._components
            else redact_url(self.settings.camera.source),
            "pose_backend": getattr(self._components.pose, "name", None)
            if self._components
            else None,
            "fps": round(state.fps, 2),
            "frame_index": state.frame_index,
            "uptime_s": round(time.time() - self.started_at, 1) if self.started_at else 0.0,
            "error": self.error,
            "last_event_seq": self.bus.last_seq,
        }
