"""Name -> experiment function registry."""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from umbral.checkpoint import Checkpoint
from umbral.config import Config
from umbral.recorder import Recorder


class Experiment(Protocol):
    def __call__(
        self,
        config: Config,
        recorder: Recorder,
        np_rng: np.random.Generator,
        resume: Checkpoint | None,
    ) -> None: ...


_REGISTRY: dict[str, Experiment] = {}


def register(name: str) -> Callable[[Experiment], Experiment]:
    def deco(fn: Experiment) -> Experiment:
        if name in _REGISTRY:
            raise ValueError(f"experiment {name!r} already registered")
        _REGISTRY[name] = fn
        return fn

    return deco


def get_experiment(name: str) -> Experiment:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown experiment {name!r}; available: {available()}") from None


def available() -> list[str]:
    return sorted(_REGISTRY)
