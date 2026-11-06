import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from typer.testing import CliRunner

from motionsense.cli import app
from motionsense.perception.yolo_pose import parse_pose_result
from motionsense.tools.evaluate import ClipResult, compute_metrics, read_labels, write_report

runner = CliRunner()


def _result(has_fall, detected, latency=None):
    return ClipResult("c.mp4", has_fall, detected, None, latency, False, {"standing": 1.0})


def test_compute_metrics():
    m = compute_metrics(
        [
            _result(True, True, 1.0),
            _result(True, True, 3.0),
            _result(True, False),
            _result(False, False),
            _result(False, True),
        ]
    )
    assert (m.tp, m.fp, m.fn, m.tn) == (2, 1, 1, 1)
    assert m.precision == round(2 / 3, 4)
    assert m.recall == round(2 / 3, 4)
    assert m.mean_latency_s == 2.0
    empty = compute_metrics([])
    assert empty.f1 == 0.0 and empty.mean_latency_s is None


def test_read_labels_and_report(tmp_path: Path):
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "video,has_fall,fall_time_s\na.mp4,1,2.5\nb.mp4,0,\n,1,\n", encoding="utf-8"
    )
    labels = read_labels(csv_path)
    assert [(Path(lbl.name).name, lbl.has_fall, lbl.fall_time) for lbl in labels] == [
        ("a.mp4", True, 2.5),
        ("b.mp4", False, None),
    ]
    report = write_report(
        tmp_path / "out",
        "t",
        compute_metrics([_result(True, True, 1.0)]),
        [_result(True, True, 1.0)],
    )
    assert "| precision | 1.0 |" in report.read_text(encoding="utf-8")


def test_cli_scenarios_lists_names():
    result = runner.invoke(app, ["scenarios"])
    assert result.exit_code == 0
    assert "fall_out_of_view" in result.output


def test_cli_eval_synthetic_reaches_perfect_score(tmp_path: Path):
    out = tmp_path / "report"
    result = runner.invoke(app, ["eval", "--synthetic", "--out", str(out)])
    assert result.exit_code == 0, result.output
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))["metrics"]
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0


def test_cli_eval_requires_exactly_one_input():
    result = runner.invoke(app, ["eval"])
    assert result.exit_code != 0


def test_parse_pose_result_with_and_without_ids():
    kxy = np.zeros((2, 17, 2))
    kconf = np.full((2, 17), 0.8)
    boxes = SimpleNamespace(
        xyxy=np.array([[0, 0, 10, 20], [5, 5, 15, 25]], float),
        conf=np.array([0.9, 0.7]),
        id=np.array([3, 4]),
        __len__=lambda: 2,
    )

    class Boxes(SimpleNamespace):
        def __len__(self):
            return 2

    res = SimpleNamespace(boxes=Boxes(**vars(boxes)), keypoints=SimpleNamespace(xy=kxy, conf=kconf))
    people = parse_pose_result(res)
    assert [p.track_id for p in people] == [3, 4]
    assert people[0].bbox == (0.0, 0.0, 10.0, 20.0)
    res_no_ids = SimpleNamespace(
        boxes=Boxes(**{**vars(boxes), "id": None}), keypoints=SimpleNamespace(xy=kxy, conf=None)
    )
    assert [p.track_id for p in parse_pose_result(res_no_ids)] == [None, None]
    assert parse_pose_result(SimpleNamespace(boxes=None, keypoints=None)) == []
