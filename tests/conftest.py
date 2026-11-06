from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from motionsense.config import Settings, load_settings
from motionsense.pipeline import Pipeline
from motionsense.simulation.backend import (
    ScriptedObjectBackend,
    ScriptedPoseBackend,
    SyntheticSource,
)
from motionsense.simulation.scenarios import SCENARIOS, Scenario
from motionsense.types import SceneState


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return load_settings(overrides={"data_dir": str(tmp_path)})


@pytest.fixture
def run_scenario(settings: Settings) -> Callable[..., tuple[Pipeline, list[SceneState]]]:
    """Run a named (or given) scenario through a fresh pipeline at 15 FPS."""

    def _run(
        scenario: str | Scenario, noise_px: float = 1.5, fps: float = 15.0, seed: int = 3
    ) -> tuple[Pipeline, list[SceneState]]:
        sc = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
        pipe = Pipeline(
            settings,
            ScriptedPoseBackend(sc, noise_px=noise_px, seed=seed),
            ScriptedObjectBackend(sc),
        )
        states = [pipe.process(frame) for frame in SyntheticSource(sc, fps=fps).frames()]
        pipe.flush(sc.duration)
        return pipe, states

    return _run
