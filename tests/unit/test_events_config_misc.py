from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from motionsense.config import Settings, load_settings
from motionsense.events.bus import EventBus
from motionsense.events.models import SCHEMA_VERSION, EventType, Severity, make_event
from motionsense.fall.context import find_support
from motionsense.render.overlay import draw_scene, encode_jpeg
from motionsense.simulation.skeleton import STAND, build_skeleton
from motionsense.tracking.ids import IouTracker, iou
from motionsense.types import EMPTY_SCENE, ObjectDetection, PersonDetection, PersonState, SceneState


class TestEvents:
    def test_make_event_defaults_and_clipping(self):
        e = make_event(EventType.FALL_CONFIRMED, ts=1.0, confidence=1.7, track_id=3)
        assert e.severity == Severity.CRITICAL
        assert e.confidence == 1.0
        d = e.to_dict()
        assert d["schema_version"] == SCHEMA_VERSION
        assert d["type"] == "fall_confirmed"
        assert "iso_time" in d

    def test_bus_assigns_sequence_and_filters_history(self):
        bus = EventBus(history_size=3)
        for i in range(5):
            bus.publish(make_event(EventType.PERSON_ENTERED, ts=float(i)))
        hist = bus.history(limit=0)
        assert [e.seq for e in hist] == [3, 4, 5]
        assert [e.seq for e in bus.history(since_seq=4)] == [5]
        assert len(bus.history(limit=2)) == 2
        assert bus.last_seq == 5
        assert bus.get(hist[0].id) == hist[0]
        assert bus.get("nope") is None

    def test_subscriber_errors_are_isolated_and_unsubscribe_works(self):
        bus = EventBus()
        received = []

        def broken(_e):
            raise RuntimeError("boom")

        bus.subscribe(broken)
        unsubscribe = bus.subscribe(received.append)
        bus.publish(make_event(EventType.PERSON_LEFT, ts=0.0))
        unsubscribe()
        unsubscribe()  # idempotent
        bus.publish(make_event(EventType.PERSON_LEFT, ts=1.0))
        assert len(received) == 1


class TestConfig:
    def test_defaults_are_valid(self):
        s = Settings()
        assert s.models.device == "auto"
        assert s.api.host == "127.0.0.1"
        assert s.models_dir.name == "models"

    def test_yaml_and_overrides_merge(self, tmp_path: Path):
        cfg = tmp_path / "c.yaml"
        cfg.write_text("camera:\n  source: 2\nfall:\n  confirm_s: 3\n", encoding="utf-8")
        s = load_settings(cfg, {"fall": {"cooldown_s": 1.0}})
        assert s.camera.source == "2"
        assert s.fall.confirm_s == 3.0
        assert s.fall.cooldown_s == 1.0

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("MOTIONSENSE_API__PORT", "9001")
        assert load_settings().api.port == 9001

    def test_precedence_overrides_env_yaml(self, monkeypatch, tmp_path: Path):
        cfg = tmp_path / "c.yaml"
        cfg.write_text("api:\n  port: 7000\n  state_hz: 5\n", encoding="utf-8")
        monkeypatch.setenv("MOTIONSENSE_API__PORT", "9001")
        s = load_settings(cfg)
        assert s.api.port == 9001  # env beats YAML
        assert s.api.state_hz == 5  # YAML beats default, other keys in the section survive
        assert load_settings(cfg, {"api": {"port": 1234}}).api.port == 1234  # CLI beats env

    def test_invalid_values_rejected(self, tmp_path: Path):
        with pytest.raises(ValidationError):
            load_settings(overrides={"api": {"port": 70000}})
        with pytest.raises(ValidationError):
            load_settings(overrides={"fall": {"unknown_key": 1}})
        bad = tmp_path / "bad.yaml"
        bad.write_text("- just\n- a list\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_settings(bad)
        with pytest.raises(FileNotFoundError):
            load_settings(tmp_path / "missing.yaml")


class TestContextAndTracking:
    def test_find_support_prefers_smallest_containing_box(self):
        objs = [
            ObjectDetection("couch", 0.9, (0, 0, 100, 100)),
            ObjectDetection("chair", 0.9, (40, 40, 60, 60)),
            ObjectDetection("tv", 0.9, (45, 45, 55, 55)),
        ]
        classes = ["couch", "chair"]
        assert find_support((50, 50), objs, classes) == "chair"
        assert find_support((10, 10), objs, classes) == "couch"
        assert find_support((500, 500), objs, classes) is None

    def test_iou(self):
        assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
        assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
        assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0

    def test_iou_tracker_keeps_ids_and_creates_new(self):
        kps, bbox = build_skeleton(STAND)
        tracker = IouTracker()
        first = tracker.assign([PersonDetection(bbox=bbox, score=0.9, keypoints=kps)])
        moved = (bbox[0] + 5, bbox[1], bbox[2] + 5, bbox[3])
        far = (0.0, 0.0, 50.0, 50.0)
        second = tracker.assign(
            [
                PersonDetection(bbox=moved, score=0.9, keypoints=kps),
                PersonDetection(bbox=far, score=0.9, keypoints=kps),
            ]
        )
        assert second[0].track_id == first[0].track_id
        assert second[1].track_id not in (None, first[0].track_id)
        tracked = [PersonDetection(bbox=bbox, score=0.9, keypoints=kps, track_id=7)]
        assert tracker.assign(tracked) is tracked


class TestOverlay:
    def test_draw_scene_does_not_modify_input(self):
        kps, bbox = build_skeleton(STAND)
        person = PersonState(
            track_id=1,
            bbox=bbox,
            keypoints=kps,
            activities={"standing": 1.0},
            top_activity="standing",
            top_confidence=1.0,
            quality="full",
            fall_state="fallen",
            fall_risk=0.9,
            reps={"push_up": 2},
        )
        scene = SceneState(
            ts=0,
            frame_index=0,
            frame_size=(1280, 720),
            fps=10,
            persons=(person,),
            objects=(ObjectDetection("chair", 0.8, (10, 10, 100, 100)),),
        )
        img = np.zeros((720, 1280, 3), np.uint8)
        out = draw_scene(img, scene)
        assert out.shape == img.shape
        assert img.sum() == 0 and out.sum() > 0
        assert encode_jpeg(out)[:2] == b"\xff\xd8"
        data = scene.to_dict()
        assert data["persons"][0]["keypoints"][0][2] == pytest.approx(0.9)
        assert "keypoints" not in scene.to_dict(include_keypoints=False)["persons"][0]
        assert EMPTY_SCENE.to_dict()["person_count"] == 0
