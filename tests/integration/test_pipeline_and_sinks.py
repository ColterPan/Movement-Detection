import json
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

from motionsense.config import load_settings
from motionsense.events.models import EventType, Severity, make_event
from motionsense.pipeline import Pipeline
from motionsense.simulation.backend import (
    ScriptedObjectBackend,
    ScriptedPoseBackend,
    SyntheticSource,
    render_background,
)
from motionsense.simulation.scenarios import BED, SCENARIOS, get_scenario
from motionsense.sinks.jsonl_log import JsonlEventLog
from motionsense.sinks.snapshot import SnapshotRecorder, is_safe_event_id
from motionsense.sinks.webhook import WebhookSink
from motionsense.sources.opencv_source import VideoFileSource
from motionsense.types import EMPTY_SCENE, Frame, PersonDetection


class _NoTrackIds:
    """Wraps a backend and strips track ids, to exercise the fallback IoU tracker."""

    name = "no-ids"

    def __init__(self, inner):
        self._inner = inner

    def infer(self, frame):
        return [
            PersonDetection(d.bbox, d.score, d.keypoints, None) for d in self._inner.infer(frame)
        ]


class _Failing:
    name = "failing"

    def infer(self, frame):
        raise RuntimeError("detector crashed")


def test_person_lifecycle_events(settings):
    sc = SCENARIOS["fall_out_of_view"]
    pipe = Pipeline(settings, ScriptedPoseBackend(sc))
    for frame in SyntheticSource(sc, fps=15).frames():
        pipe.process(frame)
    types = [e.type for e in pipe.bus.history(limit=0)]
    assert types[0] == EventType.PERSON_ENTERED
    assert EventType.PERSON_LEFT in types
    assert pipe.active_fall_states == {}


def test_backend_without_track_ids_still_works(settings):
    sc = SCENARIOS["fall"]
    pipe = Pipeline(settings, _NoTrackIds(ScriptedPoseBackend(sc)))
    for frame in SyntheticSource(sc, fps=15).frames():
        pipe.process(frame)
    assert any(e.type == EventType.FALL_CONFIRMED for e in pipe.bus.history(limit=0))


def test_object_failures_and_listener_failures_do_not_stop_pipeline(settings):
    sc = SCENARIOS["standing"]
    calls = []

    def bad_listener(frame, state):
        calls.append(state.frame_index)
        raise ValueError("listener bug")

    pipe = Pipeline(settings, ScriptedPoseBackend(sc), _Failing(), listeners=[bad_listener])
    states = [pipe.process(f) for f in SyntheticSource(sc, fps=5).frames()]
    assert len(states) == len(calls) > 0
    assert all(len(s.persons) == 1 for s in states)


def test_furniture_is_remembered_when_hidden(settings):
    sc = SCENARIOS["lie_on_bed"]

    class Flaky:
        name = "flaky"
        calls = 0

        def infer(self, frame):
            self.calls += 1
            return [BED] if self.calls == 1 else []

    pipe = Pipeline(settings, ScriptedPoseBackend(sc), Flaky())
    state = None
    for frame in SyntheticSource(sc, fps=15).frames():
        state = pipe.process(frame)
    assert state is not None and [o.label for o in state.objects] == ["bed"]


def test_objects_can_be_disabled(tmp_path):
    s = load_settings(overrides={"data_dir": str(tmp_path), "objects": {"enabled": False}})
    sc = SCENARIOS["sit_down"]
    pipe = Pipeline(s, ScriptedPoseBackend(sc), ScriptedObjectBackend(sc))
    state = pipe.process(next(iter(SyntheticSource(sc).frames())))
    assert state.objects == ()
    assert pipe.latest is state


def test_get_scenario_rejects_unknown():
    with pytest.raises(ValueError):
        get_scenario("nope")
    assert render_background(SCENARIOS["sit_down"]).shape == (720, 1280, 3)


# ------------------------------------------------------------------ sinks
def test_jsonl_log_appends_lines(tmp_path: Path):
    log = JsonlEventLog(tmp_path / "sub" / "events.jsonl")
    log(make_event(EventType.FALL_CONFIRMED, ts=1.0))
    log(make_event(EventType.PERSON_LEFT, ts=2.0))
    lines = (tmp_path / "sub" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["type"] for line in lines] == ["fall_confirmed", "person_left"]


def test_snapshot_recorder_saves_annotated_and_raw_frames(tmp_path: Path):
    rec = SnapshotRecorder(tmp_path, pre_s=1.0, post_s=1.0, hz=4.0)
    img = np.full((48, 64, 3), 128, np.uint8)
    for i in range(20):  # 2 s of frames at 10 FPS
        rec.on_frame(Frame(img, ts=i / 10, index=i), EMPTY_SCENE)
    event = make_event(EventType.FALL_CONFIRMED, ts=1.9)
    rec(event)
    rec(make_event(EventType.PERSON_ENTERED, ts=1.9))  # info events are ignored
    for i in range(20, 35):
        rec.on_frame(Frame(img, ts=i / 10, index=i), EMPTY_SCENE)

    frames = rec.frames(event.id)
    phases = [f["phase"] for f in frames]
    assert "before" in phases and "after" in phases
    assert phases == sorted(phases, key=lambda p: p == "after")  # before frames come first
    offsets = [f["offset_s"] for f in frames]
    assert offsets == sorted(offsets)
    assert all(f["raw"].endswith("_raw.jpg") for f in frames)
    assert rec.list_snapshots(event.id)[0] == "before_01.jpg"
    assert rec.snapshot_path(event.id, "before_01_raw.jpg") is not None
    assert (tmp_path / event.id / "event.json").exists()
    assert rec.event_info(event.id)["type"] == "fall_confirmed"
    assert [i["id"] for i in rec.incidents()] == [event.id]
    # path traversal and junk names are rejected
    assert rec.snapshot_path(event.id, "../../secret.jpg") is None
    assert rec.snapshot_path(event.id, "event.json") is None
    assert rec.list_snapshots("../etc") == []
    assert rec.event_info("../etc") is None
    assert not is_safe_event_id("..\\x")
    assert len(list(tmp_path.iterdir())) == 1


def test_webhook_delivers_and_retries():
    received: list[dict] = []
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(500)
        received.append(json.loads(request.content))
        return httpx.Response(200)

    sink = WebhookSink(
        ["http://hook.local/x"],
        Severity.WARNING,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    sink(make_event(EventType.PERSON_ENTERED, ts=0.0))  # below min severity
    sink(make_event(EventType.FALL_CONFIRMED, ts=1.0, confidence=0.9))
    sink.close()
    assert [r["type"] for r in received] == ["fall_confirmed"]
    assert sink.delivered == 1 and sink.failed == 0


def test_webhook_rejects_non_http_urls():
    with pytest.raises(ValueError):
        WebhookSink(["file:///etc/passwd"])


def test_webhook_counts_failures():
    sink = WebhookSink(
        ["http://hook.local/x"],
        client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503))),
    )
    sink(make_event(EventType.FALL_CONFIRMED, ts=1.0))
    sink.close()
    assert sink.failed == 1


# ------------------------------------------------------------------ sources
def test_video_file_source_reads_media_time(tmp_path: Path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 10, (64, 48))
    for i in range(12):
        writer.write(np.full((48, 64, 3), i * 10, np.uint8))
    writer.release()
    src = VideoFileSource(path)
    frames = list(src.frames())
    assert len(frames) == 12
    assert frames[5].ts == pytest.approx(0.5)
    assert src.description == "file:clip.mp4"
    src.close()
    assert list(src.frames()) == []
    with pytest.raises(FileNotFoundError):
        VideoFileSource(tmp_path / "missing.mp4")


def test_synthetic_source_loops_and_closes():
    src = SyntheticSource(SCENARIOS["standing"], fps=10, loop=True)
    frames = []
    for f in src.frames():
        frames.append(f)
        if len(frames) == 100:
            src.close()
    assert len(frames) == 100
    assert frames[-1].ts == pytest.approx(9.9)
