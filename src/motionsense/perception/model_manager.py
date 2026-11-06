"""Download model weights into the local cache and export them to OpenVINO once."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _yolo() -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "Ultralytics is not installed. Install the 'yolo' extra: uv sync --extra yolo"
        ) from exc
    return YOLO


def resolve_device(runtime: str, device: str) -> str:
    """Translate ``auto`` into the fastest available device.

    With OpenVINO, Intel integrated GPUs (e.g. Iris Xe) are typically ~10x faster than the CPU.
    """
    if device != "auto":
        return device
    if runtime == "openvino":
        try:
            import openvino as ov

            if any(d.startswith("GPU") for d in ov.Core().available_devices):
                return "intel:gpu"
        except Exception:  # pragma: no cover - depends on drivers
            logger.debug("OpenVINO GPU probe failed", exc_info=True)
        return "cpu"
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover
        return "cpu"


def warm_up(model: Any, device: str, imgsz: int, task: str) -> None:
    """Run one dummy inference so GPU kernel compilation happens at startup, not mid-stream."""
    import numpy as np

    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    try:
        model.predict(dummy, imgsz=imgsz, device=device, verbose=False)
    except Exception:
        logger.exception("warm-up of %s model failed", task)


def resolve_model(name: str, models_dir: Path, runtime: str, imgsz: int, task: str) -> Any:
    """Return a loaded Ultralytics model, downloading/exporting into ``models_dir`` as needed.

    ``name`` may be a bare model name ("yolo26n-pose"), a ``.pt`` file, or a path to an
    existing exported model directory.
    """
    yolo_cls = _yolo()
    models_dir.mkdir(parents=True, exist_ok=True)

    candidate = Path(name)
    if candidate.exists():
        logger.info("loading model from %s", candidate)
        return yolo_cls(str(candidate), task=task)

    stem = name.removesuffix(".pt")
    pt_path = models_dir / f"{stem}.pt"
    if runtime == "openvino":
        ov_dir = models_dir / f"{stem}_openvino_model"
        if not (ov_dir / "metadata.yaml").exists():
            logger.info("exporting %s to OpenVINO (one-time, ~1 min)...", stem)
            yolo_cls(str(pt_path), task=task).export(format="openvino", imgsz=imgsz, half=False)
        try:
            return yolo_cls(str(ov_dir), task=task)
        except Exception:
            logger.exception("OpenVINO model failed to load, falling back to PyTorch")
    return yolo_cls(str(pt_path), task=task)
