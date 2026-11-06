"""API tests against a real MotionService running a fast (non-realtime) synthetic scenario."""

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from motionsense.api.app import create_app
from motionsense.config import load_settings
from motionsense.service import Components, MotionService, build_components
from motionsense.simulation.backend import (
    ScriptedObjectBackend,
    ScriptedPoseBackend,
    SyntheticSource,
)
from motionsense.simulation.scenarios import SCENARIOS


def _service(tmp_path: Path, token: str | None = None) -> MotionService:
    settings = load_settings(
        overrides={"data_dir": str(tmp_path), "api": {"token": token, "stream_fps": 60}}
    )
    sc = SCENARIOS["fall"]
    comps = Components(
        SyntheticSource(sc, fps=15),
        ScriptedPoseBackend(sc, noise_px=1.0),
        ScriptedObjectBackend(sc),
    )
    return MotionService(settings, comps)


def _wait_finished(service: MotionService, timeout: float = 20.0) -> None:
    end = time.time() + timeout
    while not service.finished and time.time() < end:
        time.sleep(0.05)
    assert service.finished, service.error


@pytest.fixture
def client(tmp_path: Path):
    service = _service(tmp_path)
    with TestClient(create_app(service), base_url="http://localhost") as c:
        _wait_finished(service)
        yield c, service


def test_health_and_dashboard(client):
    c, _ = client
    health = c.get("/api/v1/health").json()
    assert health["version"] == "0.1.0"
    assert health["status"] == "finished"
    assert health["source"] == "synthetic:fall"
    page = c.get("/")
    assert page.status_code == 200 and "MotionSense" in page.text
    assert c.get("/static/app.js").status_code == 200


def test_state_endpoint(client):
    c, _ = client
    state = c.get("/api/v1/state").json()
    assert state["frame_size"] == [1280, 720]
    assert "persons" in state and "objects" in state
    assert c.get("/api/v1/state?keypoints=true").status_code == 200


def test_events_filtering_and_detail(client):
    c, _ = client
    all_events = c.get("/api/v1/events?limit=1000").json()
    assert all_events["last_seq"] >= len(all_events["events"]) > 0
    falls = c.get("/api/v1/events", params={"type": "fall_confirmed"}).json()["events"]
    assert len(falls) == 1
    event = falls[0]
    assert c.get(f"/api/v1/events/{event['id']}").json()["id"] == event["id"]
    assert c.get("/api/v1/events/doesnotexist").status_code == 404
    newer = c.get("/api/v1/events", params={"since": all_events["last_seq"]}).json()["events"]
    assert newer == []
    assert c.get("/api/v1/events?limit=0").status_code == 422


def test_snapshots_are_served(client):
    c, _ = client
    event = c.get("/api/v1/events", params={"type": "fall_confirmed"}).json()["events"][0]
    listing = c.get(event["snapshots_url"]).json()["snapshots"]
    assert listing, "expected pre/post incident snapshots"
    img = c.get(listing[0])
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    assert c.get(f"/api/v1/events/{event['id']}/snapshots/nope.jpg").status_code == 404


def test_frame_schema_and_websocket(client):
    c, _ = client
    frame = c.get("/api/v1/frame.jpg")
    assert frame.status_code == 200 and frame.content[:2] == b"\xff\xd8"
    schema = c.get("/api/v1/schema/event").json()
    assert "schema_version" in schema["properties"]
    with c.websocket_connect("ws://localhost/ws") as ws:
        kinds = {ws.receive_json()["type"] for _ in range(25)}
    assert {"event", "state"} <= kinds


def test_token_protects_endpoints(tmp_path: Path):
    service = _service(tmp_path, token="s3cret")
    with TestClient(create_app(service), base_url="http://localhost") as c:
        assert c.get("/api/v1/health").status_code == 200  # health stays public
        assert c.get("/api/v1/state").status_code == 401
        assert c.get("/api/v1/state", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert c.get("/api/v1/state", headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert c.get("/api/v1/state?token=s3cret").status_code == 200
        with pytest.raises(WebSocketDisconnect), c.websocket_connect("ws://localhost/ws") as ws:
            ws.receive_json()
        with c.websocket_connect("ws://localhost/ws?token=s3cret") as ws:
            assert ws.receive_json()["type"] in {"event", "state"}


def test_frame_unavailable_before_first_render(tmp_path: Path):
    service = _service(tmp_path)
    app = create_app(service, manage_lifecycle=False)
    with TestClient(app, base_url="http://localhost") as c:
        assert c.get("/api/v1/frame.jpg").status_code == 503
        assert (
            c.get("/api/v1/events/0123456789abcdef0123456789abcdef/snapshots").json()["snapshots"]
            == []
        )
        assert c.get("/api/v1/health").json()["status"] == "degraded"


def test_build_components_synthetic(tmp_path: Path):
    s = load_settings(overrides={"data_dir": str(tmp_path), "camera": {"source": "synthetic:walk"}})
    comps = build_components(s)
    assert comps.source.description == "synthetic:walk"
    assert comps.pose.name == "scripted-pose"
    comps.source.close()
