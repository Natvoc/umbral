from pathlib import Path

import numpy as np
import pytest

from umbral.cli import run_experiment
from umbral.config import load_config
from umbral.world.arena import Arena

BASE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


@pytest.fixture
def run_noop(tmp_path):
    """Run ``noop`` into a temp dir; returns the run directory."""

    def _run(*overrides: str) -> Path:
        cfg = load_config(BASE_CONFIG, [f"run.output_dir={tmp_path.as_posix()}",
                                        "experiment=noop", *overrides])
        return run_experiment(cfg)

    return _run


def read_metrics(run_dir: Path) -> list[str]:
    return (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()


def make_config(*overrides: str):
    return load_config(BASE_CONFIG, list(overrides))


def empty_arena(size: float = 10.0) -> Arena:
    """An arena with no objects, for hand-built scenes."""
    z2, z1 = np.zeros((0, 2)), np.zeros(0)
    return Arena(size, z2.copy(), z1.copy(), z2.copy(), z1.copy(), z2.copy(), z1.copy(),
                 np.zeros(0, bool), np.zeros(0, int))


def smooth_actions(seed: int, n: int, dim: int = 6, s: float = 0.9, scale: float = 0.8) -> list[np.ndarray]:
    """Reproducible smoothed random actions (same process as random_agent)."""
    rng = np.random.default_rng(seed)
    a = np.zeros(dim)
    out = []
    for _ in range(n):
        a = s * a + np.sqrt(1 - s * s) * scale * rng.standard_normal(dim)
        out.append(a.copy())
    return out
