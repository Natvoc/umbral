"""UmbralEnv: the 2D world plus the agent's body, with Gymnasium-style signatures.

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
"""

from __future__ import annotations

from typing import Any

import numpy as np

from umbral.body.body import ACTION_DIM, BodyParams, BodyState, limb_points, move_body, update_energy
from umbral.body.link import BodyLink
from umbral.body.sensors import ObsLayout, raw_observation, tip_food_contacts
from umbral.config import Config
from umbral.world.arena import Arena
from umbral.world.histories import build_arena, spawn_food_point


class UmbralEnv:
    action_dim = ACTION_DIM
    action_low, action_high = -1.0, 1.0

    def __init__(self, config: Config):
        self.config = config
        self.wcfg = config.world
        self.params = BodyParams.from_config(config.body)
        self.layout = ObsLayout(self.params.vision_rays)
        self.observation_dim = self.layout.dim
        self._template, self.history = build_arena(self.wcfg)
        self._world_rng: np.random.Generator | None = None
        self._link_rng: np.random.Generator | None = None
        self.arena: Arena | None = None
        self.body: BodyState | None = None
        self.link: BodyLink | None = None

    # --- Gymnasium-style API -------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        if seed is not None:
            world_ss, link_ss = np.random.SeedSequence(int(seed)).spawn(2)
            self._world_rng = np.random.default_rng(world_ss)
            self._link_rng = np.random.default_rng(link_ss)
        elif self._world_rng is None:
            raise ValueError("the first reset() needs a seed")
        self.arena = self._template.copy()
        self.body = self._start_body()
        self.link = BodyLink(self.layout, self._link_rng)
        self.steps = 0
        self.world_steps = 0
        self.food_eaten_total = 0
        self._limbs = limb_points(self.params, self.body.pos, self.body.heading, self.body.q)
        self._raw = raw_observation(self.params, self.arena, self.body, self._limbs)
        obs = self.layout.split(self.link.sense(self._raw))
        return obs, {"history_id": self.history.id, "history_name": self.history.name}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict[str, Any]]:
        intended = np.clip(np.asarray(action, dtype=float).reshape(ACTION_DIM), self.action_low, self.action_high)
        self.link.tick()
        applied = self.link.act(intended)
        frozen = bool(self.wcfg.freeze_when_disembodied) and self.link.fully_disconnected()

        eaten, balance, executed = 0, None, np.zeros(ACTION_DIM)
        if not frozen:
            executed, effort = move_body(self.params, self.arena, self.body, applied,
                                         float(self.wcfg.neutral_push_fraction))
            self._limbs = limb_points(self.params, self.body.pos, self.body.heading, self.body.q)
            self._tick_food()  # before eating: an item eaten now stays away exactly respawn_steps steps
            eaten = self._eat()
            balance = update_energy(self.params, self.body, effort, eaten)
            self._raw = raw_observation(self.params, self.arena, self.body, self._limbs)
            self.world_steps += 1
        self.steps += 1
        self.food_eaten_total += eaten

        obs = self.layout.split(self.link.sense(self._raw))
        touch = self._raw[self.layout.slices["touch"]].reshape(self.layout.shapes["touch"])
        info = {
            "action_intended": intended,
            "action_applied": applied,
            "action_executed": executed,
            "void_action": BodyLink.void_action(intended, applied),
            "frozen": frozen,
            "food_eaten": eaten,
            "contacts": int((touch[:, 0] > 0).sum()),
            "energy": self.body.energy,
            "fatigue": self.body.fatigue,
            "energy_balance": balance,
            "step": self.steps,
            "world_step": self.world_steps,
        }
        max_steps = int(self.wcfg.max_steps)
        truncated = max_steps > 0 and self.steps >= max_steps
        return obs, 0.0, False, truncated, info

    # --- internals -----------------------------------------------------------

    def _start_body(self) -> BodyState:
        p, a = self.params, self.arena
        margin = p.trunk_radius + p.reach + 0.1
        lo, hi = margin, a.size - margin
        blockers = [(a.obstacle_pos, a.obstacle_r), (a.neutral_pos, a.neutral_r)]
        pos = None
        for _ in range(int(self.wcfg.spawn_tries) * 4):
            pos = self._world_rng.uniform(lo, hi, size=2)
            if all(_clear(pos, p.trunk_radius, bp, br, 0.2) for bp, br in blockers):
                break
        heading = float(self._world_rng.uniform(-np.pi, np.pi))
        return BodyState(pos=pos, heading=heading, q=np.zeros((2, 2)), dq=np.zeros((2, 2)),
                         energy=p.energy_initial, fatigue=0.0)

    def _eat(self) -> int:
        touched = tip_food_contacts(self.params, self.arena, self._limbs)
        if len(touched):
            self.arena.food_active[touched] = False
            self.arena.food_timer[touched] = int(self.history.cfg.food_respawn_steps)
        return len(touched)

    def _tick_food(self) -> None:
        a = self.arena
        waiting = ~a.food_active
        if not waiting.any():
            return
        a.food_timer[waiting] -= 1
        trunk = (self.body.pos[None], np.array([self.params.trunk_radius + self.params.reach]))
        for j in np.flatnonzero(waiting & (a.food_timer <= 0)):
            a.food_pos[j] = spawn_food_point(self._world_rng, a, self.history, self.wcfg, extra=[trunk])
            a.food_active[j] = True
            a.food_timer[j] = 0

    # --- inspection ----------------------------------------------------------

    def snapshot(self) -> dict[str, np.ndarray]:
        """Physical state for recording and rendering (copies)."""
        return {
            "trunk_pose": np.array([*self.body.pos, self.body.heading]),
            "limb_points": self._limbs.copy(),
            "food_pos": self.arena.food_pos.copy(),
            "food_active": self.arena.food_active.copy(),
            "neutral_pos": self.arena.neutral_pos.copy(),
            "touch_intensity": self._raw[self.layout.slices["touch"]].reshape(self.layout.shapes["touch"])[:, 0].copy(),
        }

    def world_state(self) -> dict[str, Any]:
        """Everything that determines the world's future given the same actions."""
        b = self.body
        return {
            "pos": b.pos.copy(), "heading": b.heading, "q": b.q.copy(), "dq": b.dq.copy(),
            "energy": b.energy, "fatigue": b.fatigue,
            "food_pos": self.arena.food_pos.copy(), "food_active": self.arena.food_active.copy(),
            "food_timer": self.arena.food_timer.copy(), "neutral_pos": self.arena.neutral_pos.copy(),
            "rng": self._world_rng.bit_generator.state, "world_steps": self.world_steps,
        }


def _clear(p: np.ndarray, r: float, pos: np.ndarray, rad: np.ndarray, clearance: float) -> bool:
    if not len(rad):
        return True
    return bool((np.sqrt(((pos - p) ** 2).sum(1)) >= rad + r + clearance).all())


__all__ = ["UmbralEnv"]
