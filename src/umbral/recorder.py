"""Local, human-readable experiment records.

Each run gets ``<output_dir>/<YYYYMMDD-HHMMSS>_<name>/`` containing:

- ``config.yaml``   full configuration used
- ``meta.json``     seed, git commit/dirty, library versions, system info, status
- ``metrics.jsonl`` one ``{"step", "name", "value"}`` object per line, no
                    timestamps, so identical runs give byte-identical files
- ``log.txt``       timestamped human-readable log
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from umbral.config import Config

_SOURCE_DIR = Path(__file__).resolve().parent


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(_SOURCE_DIR), *args],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_info() -> dict[str, Any]:
    """Commit of the code being run, and whether the working tree had uncommitted changes."""
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
    }


def system_info() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }


class Recorder:
    """Creates the run directory and writes config, meta, metrics and log."""

    def __init__(self, config: Config, meta: dict[str, Any] | None = None):
        self.config = config
        root = Path(config.run.output_dir)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = f"{stamp}_{config.run.name}"
        run_dir = root / base
        i = 1
        while run_dir.exists():
            run_dir = root / f"{base}_{i}"
            i += 1
        run_dir.mkdir(parents=True)
        self.run_dir = run_dir

        (run_dir / "config.yaml").write_text(config.to_yaml(), encoding="utf-8")
        self.meta: dict[str, Any] = {
            "run_name": config.run.name,
            "experiment": config.experiment,
            "seed": config.seed,
            **git_info(),
            **system_info(),
            **(meta or {}),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None,
            "status": "running",
        }
        self._write_meta()
        self._metrics = open(run_dir / "metrics.jsonl", "w", encoding="utf-8", newline="\n")
        self._log = open(run_dir / "log.txt", "w", encoding="utf-8", newline="\n")
        self.log(f"run directory: {run_dir}")

    def _write_meta(self) -> None:
        text = json.dumps(self.meta, indent=2, ensure_ascii=False)
        (self.run_dir / "meta.json").write_text(text + "\n", encoding="utf-8")

    def log_metric(self, step: int, name: str, value: float) -> None:
        line = json.dumps({"step": int(step), "name": name, "value": float(value)})
        self._metrics.write(line + "\n")
        self._metrics.flush()

    def log(self, message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        self._log.write(line + "\n")
        self._log.flush()
        print(line, flush=True)

    def close(self, status: str = "completed") -> None:
        if self._metrics.closed:
            return
        self.log(f"status: {status}")
        self._metrics.close()
        self._log.close()
        self.meta["status"] = status
        self.meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
        self._write_meta()

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            self.log(f"error: {exc_type.__name__}: {exc}")
        self.close("completed" if exc is None else "failed")
