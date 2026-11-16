"""Smoothed random policy (Ornstein-Uhlenbeck-like), one generator per environment.

    a <- s * a + sqrt(1 - s^2) * scale * N(0, 1)

It never depends on anything learned (Phase 2a): the mind only watches and predicts.
Each environment has its own generator, so a row's actions do not depend on the
batch it runs in, and the state (current ``a`` plus generators) goes into
checkpoints.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np


class OUPolicy:
    def __init__(self, n_envs: int, smoothing: float, scale: float, seeds: Sequence[int], action_dim: int = 6):
        if len(seeds) != n_envs:
            raise ValueError("one seed per environment")
        self.n, self.action_dim = int(n_envs), int(action_dim)
        self.s = float(smoothing)
        self.gain = float(np.sqrt(1.0 - self.s * self.s) * scale)
        self.rngs = [np.random.default_rng(np.random.SeedSequence(int(sd))) for sd in seeds]
        self.a = np.zeros((self.n, self.action_dim))

    def act(self) -> np.ndarray:
        noise = np.stack([g.standard_normal(self.action_dim) for g in self.rngs])
        self.a = self.s * self.a + self.gain * noise
        return self.a.copy()

    def reset_rows(self, rows: Sequence[int]) -> None:
        """Start a new episode for ``rows`` (the generators continue)."""
        self.a[np.asarray(rows, int)] = 0.0

    def get_state(self) -> dict[str, Any]:
        return {"a": self.a.copy(), "rngs": [json.dumps(g.bit_generator.state) for g in self.rngs]}

    def set_state(self, state: dict[str, Any]) -> None:
        self.a = np.array(state["a"], dtype=float)
        for g, s in zip(self.rngs, state["rngs"]):
            g.bit_generator.state = json.loads(s)
