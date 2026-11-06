"""Scene context: which piece of furniture (if any) is supporting a person."""

from __future__ import annotations

from collections.abc import Iterable

from motionsense.types import BBox, ObjectDetection

LYING_SUPPORTS = frozenset({"bed", "couch"})
DEFAULT_EXPAND = 0.08


def _expand(bbox: BBox, ratio: float) -> BBox:
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return (bbox[0] - w * ratio, bbox[1] - h * ratio, bbox[2] + w * ratio, bbox[3] + h * ratio)


def _contains(bbox: BBox, point: tuple[float, float]) -> bool:
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def find_support(
    point: tuple[float, float],
    objects: Iterable[ObjectDetection],
    classes: Iterable[str],
    expand: float = DEFAULT_EXPAND,
) -> str | None:
    """Label of the smallest support object whose (slightly expanded) box contains ``point``."""
    allowed = set(classes)
    best: tuple[float, str] | None = None
    for obj in objects:
        if obj.label not in allowed:
            continue
        if not _contains(_expand(obj.bbox, expand), point):
            continue
        area = (obj.bbox[2] - obj.bbox[0]) * (obj.bbox[3] - obj.bbox[1])
        if best is None or area < best[0]:
            best = (area, obj.label)
    return best[1] if best else None
