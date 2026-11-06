"""Offline accuracy evaluation of fall detection on labelled clips.

A clip is positive if a fall happens in it. A detection is any FALL_CONFIRMED (or, with
``count_suspected``, FALL_SUSPECTED) event. Reports precision/recall/F1 and detection latency.

Labels CSV format (header required)::

    video,has_fall,fall_time_s
    clips/fall-01.mp4,1,3.2
    clips/adl-01.mp4,0,
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from motionsense.config import Settings
from motionsense.events.models import EventType
from motionsense.perception.base import ObjectBackend, PoseBackend
from motionsense.pipeline import Pipeline
from motionsense.sources.base import FrameSource


@dataclass(frozen=True, slots=True)
class ClipLabel:
    name: str
    has_fall: bool
    fall_time: float | None = None


@dataclass(frozen=True, slots=True)
class ClipResult:
    name: str
    has_fall: bool
    detected: bool
    detection_time: float | None
    latency_s: float | None
    suspected: bool
    activity_summary: dict[str, float]


@dataclass(frozen=True, slots=True)
class Metrics:
    clips: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    mean_latency_s: float | None


def compute_metrics(results: Iterable[ClipResult]) -> Metrics:
    rs = list(results)
    tp = sum(r.has_fall and r.detected for r in rs)
    fp = sum((not r.has_fall) and r.detected for r in rs)
    fn = sum(r.has_fall and not r.detected for r in rs)
    tn = sum((not r.has_fall) and not r.detected for r in rs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(rs) if rs else 0.0
    lats = [r.latency_s for r in rs if r.latency_s is not None]
    return Metrics(
        clips=len(rs),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        accuracy=round(accuracy, 4),
        mean_latency_s=round(sum(lats) / len(lats), 3) if lats else None,
    )


def any_detection_metrics(results: Iterable[ClipResult]) -> Metrics:
    """Metrics where an early FALL_SUSPECTED warning also counts as detecting the fall."""
    return compute_metrics(replace(r, detected=r.detected or r.suspected) for r in results)


def evaluate_clip(
    label: ClipLabel,
    settings: Settings,
    source: FrameSource,
    pose: PoseBackend,
    objects: ObjectBackend | None = None,
    count_suspected: bool = False,
) -> ClipResult:
    pipeline = Pipeline(settings, pose, objects)
    activity_frames: dict[str, int] = {}
    last_ts = 0.0
    for frame in source.frames():
        scene = pipeline.process(frame)
        last_ts = frame.ts
        for p in scene.persons:
            activity_frames[p.top_activity] = activity_frames.get(p.top_activity, 0) + 1
    pipeline.flush(last_ts)

    events = pipeline.bus.history(limit=0)
    wanted = {EventType.FALL_CONFIRMED} | ({EventType.FALL_SUSPECTED} if count_suspected else set())
    hits = [e for e in events if e.type in wanted]
    suspected = any(e.type == EventType.FALL_SUSPECTED for e in events)
    det_time = hits[0].ts if hits else None
    latency = (
        det_time - label.fall_time
        if det_time is not None and label.fall_time is not None and label.has_fall
        else None
    )
    total = sum(activity_frames.values()) or 1
    return ClipResult(
        name=label.name,
        has_fall=label.has_fall,
        detected=bool(hits),
        detection_time=det_time,
        latency_s=round(latency, 3) if latency is not None else None,
        suspected=suspected,
        activity_summary={k: round(v / total, 3) for k, v in sorted(activity_frames.items())},
    )


def read_labels(path: Path) -> list[ClipLabel]:
    labels: list[ClipLabel] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            video = (row.get("video") or "").strip()
            if not video:
                continue
            raw_time = (row.get("fall_time_s") or "").strip()
            fall_time = float(raw_time) if raw_time else None
            if fall_time is not None and (math.isnan(fall_time) or fall_time < 0):
                raise ValueError(f"invalid fall_time_s for {video}: {raw_time}")
            labels.append(
                ClipLabel(
                    name=str((path.parent / video).resolve())
                    if not Path(video).is_absolute()
                    else video,
                    has_fall=(row.get("has_fall") or "0").strip().lower() in {"1", "true", "yes"},
                    fall_time=fall_time,
                )
            )
    return labels


def run_evaluation(
    labels: list[ClipLabel],
    make: Callable[[ClipLabel], tuple[FrameSource, PoseBackend, ObjectBackend | None]],
    settings: Settings,
    count_suspected: bool = False,
    progress: Callable[[ClipResult], None] | None = None,
) -> tuple[Metrics, list[ClipResult]]:
    results: list[ClipResult] = []
    for label in labels:
        source, pose, objects = make(label)
        result = evaluate_clip(label, settings, source, pose, objects, count_suspected)
        results.append(result)
        if progress:
            progress(result)
    return compute_metrics(results), results


def write_report(out_dir: Path, title: str, metrics: Metrics, results: list[ClipResult]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    any_metrics = any_detection_metrics(results)
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "metrics": asdict(metrics),
                "any_detection_metrics": asdict(any_metrics),
                # File names only: reports are meant to be shared, local paths are private.
                "clips": [{**asdict(r), "name": Path(r.name).name} for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    lines = [
        f"# {title}",
        "",
        "| metric | confirmed falls | any detection (suspected or confirmed) |",
        "|---|---|---|",
        *(f"| {k} | {v} | {getattr(any_metrics, k)} |" for k, v in asdict(metrics).items()),
        "",
        "| clip | fall? | detected | latency (s) | suspected | top activities |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        top = ", ".join(
            f"{k} {v:.0%}" for k, v in sorted(r.activity_summary.items(), key=lambda kv: -kv[1])[:3]
        )
        lines.append(
            f"| {Path(r.name).name} | {'yes' if r.has_fall else 'no'} | "
            f"{'yes' if r.detected else 'no'} | {r.latency_s if r.latency_s is not None else '-'} | "
            f"{'yes' if r.suspected else 'no'} | {top} |"
        )
    report = out_dir / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
