"""Append every event as one JSON line (easy to grep, tail, or load into pandas)."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from motionsense.events.models import Event

logger = logging.getLogger(__name__)


class JsonlEventLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def __call__(self, event: Event) -> None:
        line = json.dumps(event.to_dict(), ensure_ascii=False)
        with self._lock:
            try:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:
                logger.exception("could not write event log %s", self.path)
