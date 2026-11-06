"""Regression tests for real-webcam problems: close-ups at a desk, people partly out of
view, flickering detections and furniture hallucinated on a close-up torso."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from motionsense.activity.base import ActivityContext
from motionsense.activity.rule_classifier import RuleBasedClassifier
from motionsense.features.geometry import compute_pose_features, drop_edge_keypoints, is_truncated
from motionsense.features.kinematics import compute_motion_features
from motionsense.perception.yolo_pose import parse_pose_result, resolve_tracker
from motionsense.pipeline import Pipeline
from motionsense.tracking.history import Sample, TrackHistory
from motionsense.types import KP, Frame, Keypoints, ObjectDetection, PersonDetection
from tests.unit.test_activity import STILL

FRAME = (1280, 720)


def closeup_keypoints(
    hips_on_edge: bool = True,
) -> tuple[Keypoints, tuple[float, float, float, float]]:
    """Head and shoulders filling a laptop webcam (as in a real user screenshot): the model
    still guesses hips, piled up on the bottom edge of the image."""
    xy = np.zeros((17, 2))
    conf = np.zeros(17)
    pts = {
        KP.NOSE: (500, 360),
        KP.L_EYE: (445, 318),
        KP.R_EYE: (548, 327),
        KP.L_EAR: (360, 393),
        KP.R_EAR: (594, 413),
        KP.L_SHOULDER: (178, 677),
        KP.R_SHOULDER: (746, 682),
    }
    for k, p in pts.items():
        xy[k] = p
        conf[k] = 0.9
    if hips_on_edge:
        xy[KP.L_HIP] = (300, 719)
        xy[KP.R_HIP] = (650, 719)
        conf[KP.L_HIP] = conf[KP.R_HIP] = 0.5
    return Keypoints(xy=xy, conf=conf), (17.0, 195.0, 917.0, 719.0)


def test_closeup_at_desk_is_not_lying():
    kps, bbox = closeup_keypoints()
    pose = compute_pose_features(kps, bbox, frame_size=FRAME)
    assert pose.truncated
    assert not pose.hips_visible  # the guessed hips on the border are ignored
    assert pose.posture_tilt < 20  # head above shoulders = upright
    result = RuleBasedClassifier().classify(pose, STILL, ActivityContext())
    top, _ = result.top
    assert top in {"sitting", "standing"}
    assert result.probabilities["lying"] < 0.1
    assert result.quality == "partial"


def test_without_frame_size_old_behaviour_is_kept():
    kps, bbox = closeup_keypoints(hips_on_edge=False)
    pose = compute_pose_features(kps, bbox)
    assert not pose.truncated
    assert pose.posture_tilt < 20  # head-over-shoulders still works


def test_unknown_posture_never_defaults_to_lying():
    kps = Keypoints(xy=np.zeros((17, 2)), conf=np.zeros(17))
    pose = compute_pose_features(kps, (0.0, 300.0, 1000.0, 720.0), frame_size=FRAME)
    assert np.isnan(pose.posture_tilt)  # truncated wide box: shape says nothing
    result = RuleBasedClassifier().classify(pose, STILL, ActivityContext())
    assert result.probabilities["lying"] < 0.1


def test_edge_keypoints_and_truncation():
    xy = np.full((17, 2), 500.0)
    xy[0] = (2, 300)  # left edge
    xy[1] = (640, 719)  # bottom edge
    kps = drop_edge_keypoints(Keypoints(xy=xy, conf=np.full(17, 0.9)), FRAME)
    assert kps.conf[0] == 0 and kps.conf[1] == 0 and kps.conf[2] == 0.9
    assert is_truncated((0, 0, 100, 100), FRAME)
    assert not is_truncated((100, 100, 300, 600), FRAME)
    assert not is_truncated((0, 0, 100, 100), None)


def test_moving_box_without_visible_hips_is_not_a_drop():
    """Leaning towards the camera makes the box (and its centre) jump down; that must not
    look like a fall when the hips themselves are not visible."""
    kps, _ = closeup_keypoints(hips_on_edge=False)
    hist = TrackHistory()
    for i in range(10):
        top = 100 + i * 40.0  # box slides down fast
        bbox = (17.0, top, 917.0, 719.0)
        pose = compute_pose_features(kps, bbox, frame_size=FRAME)
        hist.add(Sample(ts=i / 10, pose=pose, keypoints=kps, bbox=bbox))
    m = compute_motion_features(hist)
    assert m.drop_1s == 0.0
    assert m.peak_down_velocity == 0.0


def test_untracked_low_confidence_boxes_are_not_people():
    class Boxes(SimpleNamespace):
        def __len__(self):
            return 2

    res = SimpleNamespace(
        boxes=Boxes(
            xyxy=np.array([[0, 0, 10, 20], [5, 5, 15, 25]], float),
            conf=np.array([0.9, 0.15]),
            id=None,
        ),
        keypoints=SimpleNamespace(xy=np.zeros((2, 17, 2)), conf=np.full((2, 17), 0.8)),
    )
    assert len(parse_pose_result(res, min_untracked_score=0.4)) == 1
    assert len(parse_pose_result(res)) == 2
    assert resolve_tracker("motionsense").endswith("bytetrack_motionsense.yaml")
    assert resolve_tracker("botsort.yaml") == "botsort.yaml"


class _Script:
    """Pose backend that returns a fixed detection on chosen frames only."""

    name = "script"

    def __init__(self, det, present):
        self.det, self.present = det, present

    def infer(self, frame):
        return [self.det] if frame.index in self.present else []


def test_briefly_missed_person_stays_on_screen(settings):
    kps, _ = closeup_keypoints(hips_on_edge=False)
    det = PersonDetection(bbox=(300.0, 100.0, 600.0, 650.0), score=0.9, keypoints=kps, track_id=4)
    pipe = Pipeline(settings, _Script(det, present={0, 1, 2, 4}))
    img = np.zeros((720, 1280, 3), np.uint8)
    counts = [len(pipe.process(Frame(img, i / 10, i)).persons) for i in range(14)]
    assert counts[3] == 1  # missed for one frame: still shown
    assert counts[-1] == 0  # gone for > 0.6 s: removed
    stale = pipe.process(Frame(img, 0.5, 15)).persons
    assert stale == () or stale[0].attributes.get("stale")


def test_furniture_hallucinated_on_closeup_torso_is_dropped(settings):
    kps, bbox = closeup_keypoints(hips_on_edge=False)
    big = PersonDetection(bbox=bbox, score=0.9, keypoints=kps, track_id=1)
    on_torso = ObjectDetection("chair", 0.6, (500, 450, 700, 700))
    elsewhere = ObjectDetection("couch", 0.8, (950, 400, 1270, 700))

    class Objs:
        name = "objs"

        def infer(self, frame):
            return [on_torso, elsewhere]

    pipe = Pipeline(settings, _Script(big, present=set(range(20))), Objs())
    img = np.zeros((720, 1280, 3), np.uint8)
    pipe.process(Frame(img, 0.0, 0))  # first frame: no person known yet, everything kept
    state = pipe.process(Frame(img, 1.0, 1))  # next object pass: torso "chair" filtered
    labels = sorted(o.label for o in state.objects)
    assert labels == ["couch"]
    assert replace(on_torso, label="x").label == "x"  # dataclass sanity
