"""Time-aware exponential smoothing of activity probabilities."""

from __future__ import annotations

import math


class ProbabilitySmoother:
    """EMA whose strength depends on elapsed time, so it behaves the same at any FPS."""

    def __init__(self, tau_s: float) -> None:
        if tau_s <= 0:
            raise ValueError("tau_s must be positive")
        self._tau = tau_s
        self._state: dict[str, float] | None = None
        self._last_ts: float | None = None

    def update(self, probs: dict[str, float], ts: float) -> dict[str, float]:
        if self._state is None or self._last_ts is None:
            self._state = dict(probs)
        else:
            dt = max(ts - self._last_ts, 0.0)
            alpha = 1.0 - math.exp(-dt / self._tau)
            self._state = {
                k: (1.0 - alpha) * self._state.get(k, 0.0) + alpha * v for k, v in probs.items()
            }
        self._last_ts = ts
        total = sum(self._state.values()) or 1.0
        return {k: v / total for k, v in self._state.items()}


class StableLabel:
    """Reports a label change only after the new label has held for ``hold_s`` seconds."""

    def __init__(self, hold_s: float) -> None:
        self._hold_s = hold_s
        self.current: str | None = None
        self._candidate: str | None = None
        self._since: float = 0.0

    def update(self, label: str, ts: float) -> str | None:
        """Return the new stable label when it changes, else None."""
        if label == self.current:
            self._candidate = None
            return None
        if label != self._candidate:
            self._candidate = label
            self._since = ts
        if ts - self._since >= self._hold_s:
            self.current = label
            self._candidate = None
            return label
        return None
