"""Regression tests for robustness and security fixes found in code review."""

from dataclasses import replace
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from motionsense.api.app import _origin_allowed, _token_ok, create_app
from motionsense.config import FallConfig, load_settings
from motionsense.events.bus import EventBus
from motionsense.events.models import EventType, make_event
from motionsense.fall.detector import FallDetector, FallState
from motionsense.features.geometry import compute_pose_features
from motionsense.pipeline import Pipeline
from motionsense.service import Components, MotionService
from motionsense.simulation.backend import ScriptedPoseBackend, SyntheticSource
from motionsense.simulation.scenarios import SCENARIOS
from motionsense.simulation.skeleton import SIT, build_skeleton
from motionsense.sinks.snapshot import SnapshotRecorder
from motionsense.sinks.webhook import WebhookSink
from motionsense.sources.opencv_source import redact_url
from motionsense.tracking.ids import IouTracker
from motionsense.types import EMPTY_SCENE, Frame, PersonDetection
from tests.unit.test_fall_detector import CALM, IMPACT, LYING, POSE


# ------------------------------------------------------------------ credentials & auth
@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("rtsp://admin:hunter2@10.0.0.5:554/s1", "rtsp://***@10.0.0.5:554/s1"),
        ("http://192.168.1.4:8080/video", "http://192.168.1.4:8080/video"),
        ("0", "0"),
        (0, "0"),
        ("C:/videos/clip.mp4", "C:/videos/clip.mp4"),
    ],
)
def test_redact_url(spec, expected):
    assert redact_url(spec) == expected


def test_token_compare_handles_non_ascii():
    assert _token_ok("s3cret", "s3cret")
    assert not _token_ok("s3cret", "sécret")
    assert not _token_ok("s3cret", None)
    assert _token_ok(None, "anything")


def test_origin_rules():
    assert _origin_allowed(None, local_only=True)  # non-browser client
    assert _origin_allowed("http://localhost:8000", local_only=True)
    assert not _origin_allowed("https://evil.example", local_only=True)
    assert _origin_allowed("https://evil.example", local_only=False)  # token protects instead


def _service(tmp_path: Path, **api) -> MotionService:
    settings = load_settings(overrides={"data_dir": str(tmp_path), "api": api})
    sc = SCENARIOS["standing"]
    return MotionService(settings, Components(SyntheticSource(sc), ScriptedPoseBackend(sc), None))


def test_foreign_websites_cannot_read_local_api(tmp_path: Path):
    app = create_app(_service(tmp_path), manage_lifecycle=False)
    with TestClient(app, base_url="http://localhost") as c:
        assert c.get("/api/v1/health").status_code == 200
        rebinding = c.get("/api/v1/health", headers={"host": "attacker.example"})
        assert rebinding.status_code == 400
        with (
            pytest.raises(WebSocketDisconnect),
            c.websocket_connect(
                "ws://localhost/ws", headers={"origin": "https://evil.example"}
            ) as ws,
        ):
            ws.receive_json()
        with c.websocket_connect(
            "ws://localhost/ws", headers={"origin": "http://localhost:8000"}
        ) as ws:
            assert ws.receive_json()["type"] == "state"


def test_health_never_shows_stream_password(tmp_path: Path):
    service = _service(tmp_path)
    service.settings = load_settings(
        overrides={"data_dir": str(tmp_path), "camera": {"source": "rtsp://u:pw@cam/1"}}
    )
    service._components = None
    assert "pw" not in service.status()["source"]


# ------------------------------------------------------------------ event paging
def test_paging_forward_never_skips_events():
    bus = EventBus()
    for i in range(10):
        bus.publish(make_event(EventType.REP_COMPLETED, ts=float(i)))
    seen, since = [], 0
    while True:
        batch = bus.history(since_seq=since, limit=3)
        if not batch:
            break
        seen.extend(e.seq for e in batch)
        since = batch[-1].seq
    assert seen == list(range(1, 11))


# ------------------------------------------------------------------ resilience
class _FlakyPose:
    name = "flaky"

    def __init__(self, fail_every: int | None = None, always: bool = False):
        self.calls = 0
        self.fail_every = fail_every
        self.always = always

    def infer(self, frame):
        self.calls += 1
        if self.always or (self.fail_every and self.calls % self.fail_every == 0):
            raise RuntimeError("GPU hiccup")
        return []


def _run_service(tmp_path: Path, pose) -> MotionService:
    settings = load_settings(overrides={"data_dir": str(tmp_path)})
    service = MotionService(
        settings, Components(SyntheticSource(SCENARIOS["standing"]), pose, None)
    )
    service._run()  # synchronous for the test
    return service


def test_transient_frame_errors_do_not_stop_monitoring(tmp_path: Path):
    pose = _FlakyPose(fail_every=5)
    service = _run_service(tmp_path, pose)
    assert service.finished
    assert pose.calls == int(SCENARIOS["standing"].duration * 15)
    assert not [e for e in service.bus.history(limit=0) if e.type == EventType.SYSTEM_ERROR]


def test_persistent_errors_raise_a_critical_system_event(tmp_path: Path):
    service = _run_service(tmp_path, _FlakyPose(always=True))
    errors = [e for e in service.bus.history(limit=0) if e.type == EventType.SYSTEM_ERROR]
    assert len(errors) == 1
    assert errors[0].severity == "critical"
    assert service.status()["error"]


def test_webhook_worker_survives_unexpected_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        raise ValueError("not an httpx error")

    sink = WebhookSink(
        ["http://hook.local/x"], client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    for i in range(3):
        sink(make_event(EventType.FALL_CONFIRMED, ts=float(i)))
    sink.close()  # must not hang
    assert sink.failed == 3


# ------------------------------------------------------------------ tracking continuity
def test_id_less_frames_keep_the_backend_track_id():
    kps, bbox = build_skeleton(SIT)
    tracker = IouTracker()
    tracker.assign([PersonDetection(bbox, 0.9, kps, track_id=5)])
    out = tracker.assign([PersonDetection(bbox, 0.4, kps, track_id=None)])
    assert out[0].track_id == 5


def test_fall_state_carries_over_to_a_new_track_id(settings):
    sc = SCENARIOS["fall"]

    class SwitchIds:
        name = "switch"

        def __init__(self):
            self.inner = ScriptedPoseBackend(sc)

        def infer(self, frame):
            dets = self.inner.infer(frame)
            if 7.0 <= frame.ts < 9.0:  # person flickers out while lying on the floor
                return []
            new_id = 1 if frame.ts < 7.0 else 99
            return [replace(d, track_id=new_id) for d in dets]

    pipe = Pipeline(settings, SwitchIds())
    states = [pipe.process(f) for f in SyntheticSource(sc).frames()]
    after = [p for st in states if st.ts > 9.5 for p in st.persons]
    assert after and all(p.track_id == 99 and p.fall_state == "fallen" for p in after)
    confirmed = [e for e in pipe.bus.history(limit=0) if e.type == EventType.FALL_CONFIRMED]
    assert len(confirmed) == 1  # the incident continues; it is not re-raised


def test_climbing_into_an_undetected_chair_counts_as_recovery():
    det = FallDetector(FallConfig(confirm_s=1.0, recover_hold_s=0.5))
    det.update(0.0, POSE, IMPACT, LYING, None)
    floor_pose = replace(POSE, hip_xy=(600.0, 640.0), torso_tilt=85.0)
    for i in range(1, 15):
        det.update(i / 10, floor_pose, CALM, LYING, None)
    assert det.state == FallState.FALLEN
    seated = compute_pose_features(*build_skeleton(SIT))
    seated = replace(seated, hip_xy=(600.0, 520.0))  # hips ~0.9 torso lengths higher
    sitting = {"sitting": 0.8, "standing": 0.1, "lying": 0.1}
    signals = [
        s for i in range(15, 30) for s in det.update(i / 10, seated, CALM, sitting, None).signals
    ]
    assert [s.type for s in signals] == [EventType.FALL_RECOVERED]


# ------------------------------------------------------------------ snapshot retention
def test_snapshots_only_for_incidents_and_bounded(tmp_path: Path):
    rec = SnapshotRecorder(tmp_path, pre_s=0.5, post_s=0.0, hz=10.0, max_events=3)
    img = np.zeros((8, 8, 3), np.uint8)
    rec.on_frame(Frame(img, 0.0, 0), EMPTY_SCENE)
    rec(make_event(EventType.FALL_RECOVERED, ts=0.0))  # warning, but not an incident
    assert list(tmp_path.iterdir()) == []
    for i in range(6):
        rec(make_event(EventType.FALL_CONFIRMED, ts=float(i)))
    assert len(list(tmp_path.iterdir())) == 3
