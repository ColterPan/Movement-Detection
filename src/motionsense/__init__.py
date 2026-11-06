"""MotionSense: real-time activity recognition and fall detection from any camera.

Library usage::

    from motionsense import Pipeline, load_settings
    from motionsense.events.models import EventType

    settings = load_settings("config/default.yaml")
    pipeline = Pipeline(settings, pose_backend=my_backend)
    pipeline.bus.subscribe(lambda e: print(e.type, e.message))
    for frame in my_source.frames():
        scene = pipeline.process(frame)
"""

from motionsense.config import Settings, load_settings
from motionsense.events.bus import EventBus
from motionsense.events.models import Event, EventType, Severity
from motionsense.pipeline import Pipeline
from motionsense.types import (
    Frame,
    Keypoints,
    ObjectDetection,
    PersonDetection,
    PersonState,
    SceneState,
)

__version__ = "0.1.0"

__all__ = [
    "Event",
    "EventBus",
    "EventType",
    "Frame",
    "Keypoints",
    "ObjectDetection",
    "PersonDetection",
    "PersonState",
    "Pipeline",
    "SceneState",
    "Settings",
    "Severity",
    "__version__",
    "load_settings",
]
