"""Frame source interface."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from motionsense.types import Frame


class FrameSource(Protocol):
    """Yields frames until exhausted (files) or closed (live cameras)."""

    description: str
    is_live: bool

    def frames(self) -> Iterator[Frame]: ...

    def close(self) -> None: ...
