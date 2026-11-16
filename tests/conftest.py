from pathlib import Path

import pytest

from umbral.cli import run_experiment
from umbral.config import load_config

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
