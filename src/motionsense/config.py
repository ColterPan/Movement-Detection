"""Validated configuration.

Precedence (highest first): explicit overrides > environment (``MOTIONSENSE_<SECTION>__<KEY>``)
> YAML file > defaults defined here.
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


def default_data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME") or "~/.local/share"
    return Path(base).expanduser() / "motionsense"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CameraConfig(_Section):
    source: str = Field(
        "0",
        description="Webcam index, video path, rtsp:// or http:// URL, or 'synthetic[:scenario]'",
    )
    width: int = Field(1280, ge=160, le=7680)
    height: int = Field(720, ge=120, le=4320)
    fps: int = Field(30, ge=1, le=120)
    camera_id: str = "cam0"
    realtime_files: bool = Field(True, description="Play video files at their native speed")

    @field_validator("source", mode="before")
    @classmethod
    def _source_to_str(cls, v: Any) -> str:
        return str(v)


class ModelConfig(_Section):
    pose_model: str = "yolo26s-pose"
    object_model: str = "yolo26n"
    runtime: Literal["openvino", "torch"] = "openvino"
    device: str = Field(
        "auto", description="'auto', 'cpu', 'intel:gpu' (OpenVINO iGPU), 'cuda:0'..."
    )
    imgsz: int = Field(640, ge=160, le=1920)
    pose_conf: float = Field(0.4, ge=0.05, le=0.95)
    object_conf: float = Field(0.4, ge=0.05, le=0.95)
    tracker: str = "bytetrack.yaml"
    keypoint_min_conf: float = Field(0.35, ge=0.0, le=1.0)


class ObjectsConfig(_Section):
    enabled: bool = True
    interval_s: float = Field(0.5, gt=0.0, description="Run object detection at most this often")
    support_classes: tuple[str, ...] = ("bed", "couch", "chair", "bench", "toilet")


class ActivityConfig(_Section):
    ema_tau_s: float = Field(0.45, gt=0.0)
    change_hold_s: float = Field(
        1.0, ge=0.0, description="Activity must hold this long before ACTIVITY_CHANGED"
    )
    looking_around_yaw_std: float = Field(0.28, gt=0.0)


class FallConfig(_Section):
    drop_threshold: float = Field(0.7, gt=0.0, description="Hip drop in torso lengths within 1 s")
    tilt_change_threshold: float = Field(
        40.0, gt=0.0, description="Torso rotation in degrees within 1.5 s"
    )
    velocity_threshold: float = Field(
        1.3, gt=0.0, description="Peak downward hip speed, torso lengths/s"
    )
    impact_score_threshold: float = Field(0.5, gt=0.0, lt=1.0)
    confirm_s: float = Field(2.0, gt=0.0, description="Stay down this long after impact to confirm")
    recover_hold_s: float = Field(1.0, ge=0.0, description="Upright this long counts as recovered")
    inactive_alert_s: float = Field(
        30.0, gt=0.0, description="Still down this long after a fall -> escalation"
    )
    furniture_factor: float = Field(
        0.35, ge=0.0, le=1.0, description="Impact multiplier when landing on bed/couch"
    )
    cooldown_s: float = Field(
        3.0, ge=0.0, description="Min seconds between FALL_SUSPECTED for one person"
    )
    track_lost_s: float = Field(1.5, gt=0.0)
    emit_suspected: bool = True


class ApiConfig(_Section):
    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)
    token: str | None = Field(None, description="If set, required as Bearer token or ?token=")
    state_hz: float = Field(10.0, gt=0.0, le=60.0)
    stream_fps: float = Field(15.0, gt=0.0, le=60.0)
    jpeg_quality: int = Field(75, ge=10, le=100)


class SinksConfig(_Section):
    jsonl: bool = True
    snapshots: bool = True
    snapshot_pre_s: float = Field(2.0, ge=0.0)
    snapshot_post_s: float = Field(2.0, ge=0.0)
    snapshot_hz: float = Field(2.0, gt=0.0)
    snapshot_max_events: int = Field(
        200, ge=1, description="Keep at most this many incidents on disk"
    )
    webhooks: tuple[str, ...] = ()
    webhook_min_severity: Literal["info", "warning", "critical"] = "warning"


_YAML_DATA: ContextVar[dict[str, Any] | None] = ContextVar("motionsense_yaml_data", default=None)


class _YamlSource(PydanticBaseSettingsSource):
    """Settings source for the YAML file passed to ``load_settings``."""

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        return (_YAML_DATA.get() or {}).get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return dict(_YAML_DATA.get() or {})


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MOTIONSENSE_", env_nested_delimiter="__", extra="forbid", frozen=True
    )

    camera: CameraConfig = Field(default_factory=CameraConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    objects: ObjectsConfig = Field(default_factory=ObjectsConfig)
    activity: ActivityConfig = Field(default_factory=ActivityConfig)
    fall: FallConfig = Field(default_factory=FallConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    sinks: SinksConfig = Field(default_factory=SinksConfig)
    data_dir: Path = Field(default_factory=default_data_dir)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Highest priority first: explicit (CLI) overrides > environment > YAML file > defaults.
        return init_settings, env_settings, _YamlSource(settings_cls), file_secret_settings

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def events_dir(self) -> Path:
        return self.data_dir / "events"


def load_settings(
    path: str | Path | None = None, overrides: dict[str, Any] | None = None
) -> Settings:
    """Load settings from an optional YAML file plus nested ``overrides``.

    Precedence (highest first): ``overrides`` > ``MOTIONSENSE_*`` environment > YAML > defaults.
    """
    data: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"config file not found: {p}")
        loaded = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"config file must contain a mapping at the top level: {p}")
        data = loaded
    token = _YAML_DATA.set(data)
    try:
        return Settings(**(overrides or {}))
    finally:
        _YAML_DATA.reset(token)
