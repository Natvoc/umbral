"""UmbralEnv: one world plus the agent's body, with Gymnasium-style signatures.

    env = UmbralEnv(config)
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)

``obs`` is a dict with one array per channel (``proprio`` (10,), ``touch``
(10, 6), ``vision`` (7, 5), ``intero`` (2,)) plus ``flat`` (107,), the
concatenation in that order. The layout is documented in
``umbral.body.sensors``. Everything the mind sees has passed through
``env.link`` (a ``BodyLink``).

``reward`` is always 0.0, by design: drives (curiosity, caution) live in the
mind, not in the environment. ``terminated`` is always False (there is no
death). ``truncated`` becomes True after ``world.max_steps`` steps if that is > 0.

Disembodiment freeze: with ``world.freeze_when_disembodied`` and every link
segment disconnected (cut, noise or a finished fade), the world and the physical
body are paused: no metabolism, no food respawn, nothing moves. The mind still
gets observations (through the link) and its intended actions are still
recorded. When the body is restored the agent finds it exactly as it left it.

Randomness: ``reset(seed)`` derives two independent generators from ``seed``:
one for the world (start pose, food respawns) and one for the link's noise.

Implementation: a ``BatchedUmbralEnv`` of size one. ``env.arena``, ``env.body``
and ``env.link`` are live views into it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from umbral.body.body import ACTION_DIM
from umbral.body.link import BodyLink
from umbral.config import Config
from umbral.world.arena import Arena
from umbral.world.batched import BatchedUmbralEnv


class _BodyView:
    """Live view of the body of a batch row (same attributes as ``BodyState``)."""

    def __init__(self, env: BatchedUmbralEnv, row: int):
        self._e, self._r = env, row

    pos = property(lambda self: self._e.pos[self._r])
    heading = property(lambda self: float(self._e.heading[self._r]))
    q = property(lambda self: self._e.q[self._r])
    dq = property(lambda self: self._e.dq[self._r])
    energy = property(lambda self: float(self._e.energy[self._r]))
    fatigue = property(lambda self: float(self._e.fatigue[self._r]))


class UmbralEnv:
    action_dim = ACTION_DIM
    action_low, action_high = -1.0, 1.0

    def __init__(self, config: Config, history_id: str | int | None = None):
        self.config = config
        hid = config.world.history_id if history_id is None else history_id
        self.batch = BatchedUmbralEnv(config, 1, [hid])
        self.params = self.batch.params
        self.layout = self.batch.layout
        self.observation_dim = self.batch.observation_dim
        self.history = self.batch.histories[0]
        self.link = BodyLink(self.layout, batch=self.batch.link, row=0)
        self.body = _BodyView(self.batch, 0)

    # --- live views ------------------------------------------------------------

    @property
    def arena(self) -> Arena:
        return self.batch.row_arena(0)

    @property
    def _limbs(self) -> np.ndarray:
        return self.batch.limbs[0]

    @property
    def _raw(self) -> np.ndarray:
        return self.batch.raw[0]

    @property
    def steps(self) -> int:
        return int(self.batch.steps[0])

    @property
    def world_steps(self) -> int:
        return int(self.batch.world_steps[0])

    @property
    def food_eaten_total(self) -> int:
        return int(self.batch.food_eaten_total[0])

    # --- Gymnasium-style API -------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        obs, _ = self.batch.reset(seed=[seed], options=options)
        return {k: v[0] for k, v in obs.items()}, {"history_id": self.history.id, "history_name": self.history.name}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict[str, Any]]:
        obs, _, _, truncated, b = self.batch.step(np.asarray(action, dtype=float).reshape(1, ACTION_DIM))
        frozen = bool(b["frozen"][0])
        bal = b["energy_balance"]
        info = {
            "action_intended": b["action_intended"][0],
            "action_applied": b["action_applied"][0],
            "action_executed": b["action_executed"][0],
            "void_action": {k: float(v[0]) for k, v in b["void_action"].items()},
            "frozen": frozen,
            "food_eaten": int(b["food_eaten"][0]),
            "contacts": int(b["contacts"][0]),
            "energy": float(b["energy"][0]),
            "fatigue": float(b["fatigue"][0]),
            "energy_balance": None if (bal is None or frozen) else {
                k: float(bal[k][0]) for k in ("before", "basal", "effort_cost", "food_gain", "unclipped", "after")},
            "step": int(b["step"][0]),
            "world_step": int(b["world_step"][0]),
        }
        return {k: v[0] for k, v in obs.items()}, 0.0, False, bool(truncated[0]), info

    # --- inspection ----------------------------------------------------------

    def snapshot(self) -> dict[str, np.ndarray]:
        """Physical state for recording and rendering (copies, real objects only)."""
        a = self.arena
        touch = self._raw[self.layout.slices["touch"]].reshape(self.layout.shapes["touch"])
        return {
            "trunk_pose": np.array([*self.body.pos, self.body.heading]),
            "limb_points": self._limbs.copy(),
            "food_pos": a.food_pos.copy(),
            "food_active": a.food_active.copy(),
            "neutral_pos": a.neutral_pos.copy(),
            "touch_intensity": touch[:, 0].copy(),
        }

    def world_state(self) -> dict[str, Any]:
        """Everything that determines the world's future given the same actions."""
        return self.batch.world_state(0)

    def get_state(self) -> dict[str, Any]:
        return self.batch.get_state()

    def set_state(self, state: dict[str, Any]) -> None:
        self.batch.set_state(state)


__all__ = ["UmbralEnv", "BatchedUmbralEnv"]
