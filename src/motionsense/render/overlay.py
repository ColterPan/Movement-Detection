"""Draw skeletons, labels, activity bars and fall alerts onto frames."""

from __future__ import annotations

import cv2
import numpy as np

from motionsense.types import SKELETON_EDGES, PersonState, SceneState

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (245, 245, 245)
GREY = (150, 150, 150)
OBJECT_COLOR = (200, 170, 90)
RISK_OK = (90, 200, 90)
RISK_WARN = (40, 170, 255)
RISK_BAD = (60, 60, 235)
KP_MIN_CONF = 0.35

ACTIVITY_COLORS: dict[str, tuple[int, int, int]] = {
    "standing": (120, 200, 90),
    "sitting": (220, 170, 60),
    "lying": (200, 120, 200),
    "walking": (90, 200, 220),
    "running": (60, 140, 250),
    "squatting": (230, 220, 90),
    "push_up": (250, 120, 120),
}


def _risk_color(risk: float) -> tuple[int, int, int]:
    if risk >= 0.6:
        return RISK_BAD
    if risk >= 0.3:
        return RISK_WARN
    return RISK_OK


def _label(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.55,
) -> None:
    (w, h), base = cv2.getTextSize(text, FONT, scale, 1)
    x, y = org
    cv2.rectangle(img, (x, y - h - base - 4), (x + w + 8, y + 2), (25, 25, 25), -1)
    cv2.putText(img, text, (x + 4, y - base), FONT, scale, color, 1, cv2.LINE_AA)


def draw_person(img: np.ndarray, person: PersonState) -> None:
    color = ACTIVITY_COLORS.get(person.top_activity, WHITE)
    alarm = person.fall_state in ("fallen", "inactive")
    box_color = RISK_BAD if alarm else color
    x1, y1, x2, y2 = (int(v) for v in person.bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 3 if alarm else 2)

    xy, conf = person.keypoints.xy, person.keypoints.conf
    for a, b in SKELETON_EDGES:
        if conf[a] >= KP_MIN_CONF and conf[b] >= KP_MIN_CONF:
            pa = (int(xy[a][0]), int(xy[a][1]))
            pb = (int(xy[b][0]), int(xy[b][1]))
            cv2.line(img, pa, pb, color, 2, cv2.LINE_AA)
    for (x, y), c in zip(xy, conf, strict=True):
        if c >= KP_MIN_CONF:
            cv2.circle(img, (int(x), int(y)), 3, WHITE, -1, cv2.LINE_AA)

    title = f"#{person.track_id} {person.top_activity.replace('_', ' ')} {person.top_confidence * 100:.0f}%"
    _label(img, title, (x1, max(y1, 22)), color)
    risk_text = f"fall risk {person.fall_risk * 100:.0f}%"
    if person.fall_state != "monitoring":
        risk_text += f" [{person.fall_state.upper()}]"
    _label(img, risk_text, (x1, max(y1, 22) + 24), _risk_color(person.fall_risk), 0.5)
    reps = ", ".join(f"{k.replace('_', ' ')} x{v}" for k, v in person.reps.items() if v)
    if reps:
        _label(img, reps, (x1, max(y1, 22) + 46), WHITE, 0.5)

    if alarm:
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), RISK_BAD, 8)
        _label(img, f"FALL DETECTED - PERSON #{person.track_id}", (w // 2 - 170, 40), WHITE, 0.8)


def draw_objects(img: np.ndarray, scene: SceneState) -> None:
    for obj in scene.objects:
        x1, y1, x2, y2 = (int(v) for v in obj.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), OBJECT_COLOR, 1)
        cv2.putText(
            img,
            f"{obj.label} {obj.score:.2f}",
            (x1 + 3, y1 + 14),
            FONT,
            0.45,
            OBJECT_COLOR,
            1,
            cv2.LINE_AA,
        )


def draw_scene(image: np.ndarray, scene: SceneState) -> np.ndarray:
    """Return an annotated copy of ``image`` (the input is not modified)."""
    img = image.copy()
    draw_objects(img, scene)
    for person in scene.persons:
        draw_person(img, person)
    _label(
        img, f"{scene.fps:4.1f} FPS  people: {len(scene.persons)}", (8, img.shape[0] - 8), GREY, 0.5
    )
    return img


def encode_jpeg(image: np.ndarray, quality: int = 75) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:  # pragma: no cover - only fails on invalid images
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()
