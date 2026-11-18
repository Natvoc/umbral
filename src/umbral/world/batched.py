"""BatchedUmbralEnv: N worlds simulated at once with the same vectorised kernels.

    env = BatchedUmbralEnv(config, n_envs=16, history_ids=["0", "1", "2", "3"] * 4)
    obs, info = env.reset(seed=[s0, s1, ...])          # one seed per environment
    obs, reward, terminated, truncated, info = env.step(actions)   # actions (N, 6)

Every array has a leading batch axis. Observations are dicts of channel arrays
``proprio (N,10)``, ``touch (N,10,6)``, ``vision (N,7,5)``, ``intero (N,2)`` and
``flat (N,107)`` (layout in ``umbral.body.sensors``). ``reward`` is always 0,
``terminated`` always False; ``truncated`` follows ``world.max_steps``.

Exact equivalence: environment ``i`` of a batch follows the same trajectory as a
batch of one (``UmbralEnv``) with the same seed, history and actions. To that
end:

- objects are padded to a fixed capacity per kind (the maximum over every history
  in ``world.histories``), so each row has the same shapes whatever the batch;
  padding objects sit far away with radius 0 and are switched off;
- kernels compute rows independently (elementwise ops, last-axis reductions, no
  matrix products);
- each environment owns its world and link generators (derived from its seed),
  and only consumes them when it needs to (respawns, resets, link noise).

Each environment may have its own history and its own BodyLink (``env.link`` is a
``BatchedBodyLink``; ``env.link.set_mode(..., envs=[i])``).
"""

from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np

from umbral.body.body import ACTION_DIM, BodyParams, limb_points_b, move_body_b, update_energy_b
from umbral.body.link import BatchedBodyLink
from umbral.body.sensors import ObsLayout, raw_observation_b, tip_food_b
from umbral.config import Config
from umbral.world.arena import FAR, FOOD, NEUTRAL, OBSTACLE, Arena
from umbral.world.histories import History, build_arena, spawn_food_point

_STATE_ARRAYS = ("obstacle_pos", "obstacle_r", "obstacle_on", "neutral_pos", "neutral_r", "neutral_on",
                 "food_pos", "food_r", "food_valid", "food_active", "food_timer",
                 "pos", "heading", "q", "dq", "energy", "fatigue", "limbs", "raw",
                 "steps", "world_steps", "food_eaten_total", "metabolism")


def _pad(arr: np.ndarray, n: int, fill: float) -> np.ndarray:
    out = np.full((n, *arr.shape[1:]), fill, dtype=arr.dtype)
    out[: len(arr)] = arr
    return out


class BatchedUmbralEnv:
    action_dim = ACTION_DIM
    action_low, action_high = -1.0, 1.0

    def __init__(self, config: Config, n_envs: int, history_ids: Sequence[str | int] | None = None):
        self.config = config
        self.wcfg = config.world
        self.params = BodyParams.from_config(config.body)
        self.layout = ObsLayout(self.params.vision_rays)
        self.observation_dim = self.layout.dim
        self.n = int(n_envs)
        if history_ids is None:
            history_ids = [self.wcfg.history_id] * self.n
        if len(history_ids) != self.n:
            raise ValueError("history_ids must have one entry per environment")
        self.history_ids = [str(h) for h in history_ids]
        self.size = float(self.wcfg.size)
        self.freeze = bool(self.wcfg.freeze_when_disembodied)
        self.push_fraction = float(self.wcfg.neutral_push_fraction)
        self.max_steps = int(self.wcfg.max_steps)

        # Templates for every defined history fix the padding capacity.
        self._templates: dict[str, tuple[Arena, History]] = {
            hid: build_arena(self.wcfg, hid) for hid in sorted(self.wcfg.histories)}
        arenas = [a for a, _ in self._templates.values()]
        self.cap_obstacles = max(len(a.obstacle_r) for a in arenas)
        # One extra neutral slot, always off in normal life: the probe used by the behaviour
        # battery (an object that appears suddenly). It is padding like any other until placed.
        self.cap_neutrals = max(len(a.neutral_r) for a in arenas) + 1
        self.probe_slot = self.cap_neutrals - 1
        self.cap_food = max(len(a.food_r) for a in arenas)
        self.histories = [self._templates[h][1] for h in self.history_ids]
        self.respawn_steps = np.array([int(h.cfg.food_respawn_steps) for h in self.histories])
        self.obj_kind = np.concatenate([np.full(self.cap_food, FOOD), np.full(self.cap_obstacles, OBSTACLE),
                                        np.full(self.cap_neutrals, NEUTRAL)])

        B, O, N, F = self.n, self.cap_obstacles, self.cap_neutrals, self.cap_food
        self.obstacle_pos, self.obstacle_r = np.full((B, O, 2), FAR), np.zeros((B, O))
        self.obstacle_on = np.zeros((B, O), bool)
        self.neutral_pos, self.neutral_r = np.full((B, N, 2), FAR), np.zeros((B, N))
        self.neutral_on = np.zeros((B, N), bool)
        self.food_pos, self.food_r = np.full((B, F, 2), FAR), np.zeros((B, F))
        self.food_valid = np.zeros((B, F), bool)
        self.food_active = np.zeros((B, F), bool)
        self.food_timer = np.zeros((B, F), np.int64)
        self.pos, self.heading = np.zeros((B, 2)), np.zeros(B)
        self.q, self.dq = np.zeros((B, 2, 2)), np.zeros((B, 2, 2))
        self.energy, self.fatigue = np.zeros(B), np.zeros(B)
        self.limbs = np.zeros((B, 2, 3, 2))
        self.raw = np.zeros((B, self.layout.dim))
        self.steps = np.zeros(B, np.int64)
        self.world_steps = np.zeros(B, np.int64)
        self.food_eaten_total = np.zeros(B, np.int64)
        self.world_rngs: list[np.random.Generator | None] = [None] * B
        # Factor on the energy cost of living (basal + effort), the same for every row. It is 1
        # except in the protected childhood of an agent's life (set by umbral.mind.life).
        self.metabolism = np.ones(())
        self.link = BatchedBodyLink(self.layout, [np.random.default_rng(0) for _ in range(B)])
        self._seeded = np.zeros(B, bool)

    # --- Gymnasium-style API -------------------------------------------------

    def reset(self, seed: int | Sequence[int | None] | None = None, options: dict | None = None,
              envs: Sequence[int] | None = None) -> tuple[dict, dict]:
        """Reset all environments (or only ``envs``).

        ``seed``: one per reset environment, or an int ``s`` meaning ``s + i`` for
        the i-th reset environment, or None to continue each one's generators.
        Returns the observation of *all* environments (rows not reset are
        returned unchanged, as last observed).
        """
        rows = np.arange(self.n) if envs is None else np.asarray(envs, int)
        if seed is None:
            seeds: list[int | None] = [None] * len(rows)
        elif isinstance(seed, (int, np.integer)):
            seeds = [int(seed) + i for i in range(len(rows))]
        else:
            seeds = list(seed)
            if len(seeds) != len(rows):
                raise ValueError("one seed per reset environment")
        for r, s in zip(rows, seeds):
            if s is not None:
                world_ss, link_ss = np.random.SeedSequence(int(s)).spawn(2)
                self.world_rngs[r] = np.random.default_rng(world_ss)
                self.link.rngs[r] = np.random.default_rng(link_ss)
                self._seeded[r] = True
            elif not self._seeded[r]:
                raise ValueError("the first reset() of each environment needs a seed")
            self._reset_row(int(r))
        self.link.reset_rows(rows)
        self.limbs[rows] = limb_points_b(self.params, self.pos[rows], self.heading[rows], self.q[rows])
        self.raw[rows] = self._raw_rows(rows)
        self._last_obs = getattr(self, "_last_obs", np.zeros((self.n, self.layout.dim)))
        self._last_obs[rows] = self.link.sense(self.raw[rows], rows=rows)
        info = {"history_id": list(self.history_ids), "history_name": [h.name for h in self.histories]}
        return self.layout.split(self._last_obs), info

    def step(self, actions: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        p = self.params
        intended = np.clip(np.asarray(actions, dtype=float).reshape(self.n, ACTION_DIM),
                           self.action_low, self.action_high)
        self.link.tick()
        applied = self.link.act(intended)
        frozen = self.link.fully_disconnected() if self.freeze else np.zeros(self.n, bool)
        live = ~frozen

        eaten = np.zeros(self.n, np.int64)
        executed = np.zeros((self.n, ACTION_DIM))
        balance = None
        if live.any():
            mv = move_body_b(p, self.size, self.pos, self.heading, self.q, self.energy, self.fatigue,
                             self.obstacle_pos, self.obstacle_r, self.neutral_pos, self.neutral_r,
                             applied, self.push_fraction)
            L = live
            self.pos[L], self.heading[L], self.q[L], self.dq[L] = mv["pos"][L], mv["heading"][L], mv["q"][L], mv["dq"][L]
            self.neutral_pos[L] = mv["neutral_pos"][L]
            executed[L] = mv["executed"][L]
            self.limbs[L] = limb_points_b(p, self.pos[L], self.heading[L], self.q[L])
            self._tick_food(L)  # before eating: an item eaten now stays away exactly respawn_steps steps
            touched = tip_food_b(p, self.limbs, self.food_pos, self.food_r, self.food_active) & L[:, None]
            self.food_active[touched] = False
            self.food_timer[touched] = np.broadcast_to(self.respawn_steps[:, None], touched.shape)[touched]
            eaten = touched.sum(axis=1)
            balance = update_energy_b(p, self.energy, self.fatigue, mv["effort"], eaten.astype(float),
                                      float(self.metabolism))
            self.energy[L], self.fatigue[L] = balance["after"][L], balance["fatigue"][L]
            rows = np.flatnonzero(L)
            self.raw[rows] = self._raw_rows(rows)
            self.world_steps[L] += 1
        self.steps += 1
        self.food_eaten_total += eaten

        self._last_obs = self.link.sense(self.raw)
        touch = self.raw[:, self.layout.slices["touch"]].reshape(self.n, *self.layout.shapes["touch"])
        info = {
            "action_intended": intended, "action_applied": applied, "action_executed": executed,
            "void_action": self.link.void_action(intended, applied),
            "frozen": frozen, "food_eaten": eaten, "contacts": (touch[:, :, 0] > 0).sum(axis=1),
            "energy": self.energy.copy(), "fatigue": self.fatigue.copy(),
            "energy_balance": balance, "step": self.steps.copy(), "world_step": self.world_steps.copy(),
        }
        truncated = (self.steps >= self.max_steps) if self.max_steps > 0 else np.zeros(self.n, bool)
        return (self.layout.split(self._last_obs), np.zeros(self.n), np.zeros(self.n, bool), truncated, info)

    # --- internals -----------------------------------------------------------

    def _raw_rows(self, rows: np.ndarray) -> np.ndarray:
        obj_pos = np.concatenate([self.food_pos[rows], self.obstacle_pos[rows], self.neutral_pos[rows]], axis=1)
        obj_r = np.concatenate([self.food_r[rows], self.obstacle_r[rows], self.neutral_r[rows]], axis=1)
        obj_on = np.concatenate([self.food_active[rows], self.obstacle_on[rows], self.neutral_on[rows]], axis=1)
        return raw_observation_b(self.params, self.size, self.pos[rows], self.heading[rows], self.q[rows],
                                 self.dq[rows], self.energy[rows], self.fatigue[rows], self.limbs[rows],
                                 obj_pos, obj_r, self.obj_kind, obj_on)

    def _reset_row(self, r: int) -> None:
        tpl, _ = self._templates[self.history_ids[r]]
        O, N, F = self.cap_obstacles, self.cap_neutrals, self.cap_food
        self.obstacle_pos[r] = _pad(tpl.obstacle_pos, O, FAR)
        self.obstacle_r[r] = _pad(tpl.obstacle_r, O, 0.0)
        self.obstacle_on[r] = np.arange(O) < len(tpl.obstacle_r)
        self.neutral_pos[r] = _pad(tpl.neutral_pos, N, FAR)
        self.neutral_r[r] = _pad(tpl.neutral_r, N, 0.0)
        self.neutral_on[r] = np.arange(N) < len(tpl.neutral_r)
        self.food_pos[r] = _pad(tpl.food_pos, F, FAR)
        self.food_r[r] = _pad(tpl.food_r, F, 0.0)
        self.food_valid[r] = np.arange(F) < len(tpl.food_r)
        self.food_active[r] = self.food_valid[r] & _pad(tpl.food_active, F, False)
        self.food_timer[r] = 0
        self._start_body(r)
        self.steps[r] = self.world_steps[r] = self.food_eaten_total[r] = 0

    def _start_body(self, r: int) -> None:
        p, rng = self.params, self.world_rngs[r]
        margin = p.trunk_radius + p.reach + 0.1
        lo, hi = margin, self.size - margin
        on_o, on_n = self.obstacle_on[r], self.neutral_on[r]
        blockers = [(self.obstacle_pos[r][on_o], self.obstacle_r[r][on_o]),
                    (self.neutral_pos[r][on_n], self.neutral_r[r][on_n])]
        pos = None
        for _ in range(int(self.wcfg.spawn_tries) * 4):
            pos = rng.uniform(lo, hi, size=2)
            if all(_clear(pos, p.trunk_radius, bp, br, 0.2) for bp, br in blockers):
                break
        self.pos[r] = pos
        self.heading[r] = float(rng.uniform(-np.pi, np.pi))
        self.q[r] = 0.0
        self.dq[r] = 0.0
        self.energy[r] = p.energy_initial
        self.fatigue[r] = 0.0

    def place_probe(self, r: int, pos: np.ndarray, radius: float) -> None:
        """Switch on the probe neutral object of environment ``r`` at ``pos`` (experimenter's action)."""
        j = self.probe_slot
        self.neutral_pos[r, j] = pos
        self.neutral_r[r, j] = radius
        self.neutral_on[r, j] = True
        self.raw[r] = self._raw_rows(np.array([r]))[0]

    def clear_probe(self, r: int) -> None:
        """Switch the probe of environment ``r`` off again (back to padding)."""
        j = self.probe_slot
        self.neutral_pos[r, j] = FAR
        self.neutral_r[r, j] = 0.0
        self.neutral_on[r, j] = False
        self.raw[r] = self._raw_rows(np.array([r]))[0]

    def row_arena(self, r: int) -> Arena:
        """Views of environment ``r``'s objects, trimmed to the real (non-padding) ones."""
        no = int(self.obstacle_on[r].sum())
        nn = int(self.neutral_on[r, :self.probe_slot].sum())
        nf = int(self.food_valid[r].sum())
        if self.neutral_on[r, self.probe_slot]:  # the probe, when present, is a real object (copies)
            keep = np.r_[np.arange(nn), self.probe_slot]
            n_pos, n_r = self.neutral_pos[r, keep], self.neutral_r[r, keep]
        else:
            n_pos, n_r = self.neutral_pos[r, :nn], self.neutral_r[r, :nn]
        return Arena(self.size, self.obstacle_pos[r, :no], self.obstacle_r[r, :no], n_pos, n_r,
                     self.food_pos[r, :nf], self.food_r[r, :nf], self.food_active[r, :nf], self.food_timer[r, :nf])

    def _tick_food(self, live: np.ndarray) -> None:
        waiting = self.food_valid & ~self.food_active & live[:, None]
        if not waiting.any():
            return
        self.food_timer[waiting] -= 1
        ready = waiting & (self.food_timer <= 0)
        for r in np.flatnonzero(ready.any(axis=1)):
            arena = self.row_arena(r)
            trunk = (self.pos[r][None], np.array([self.params.trunk_radius + self.params.reach]))
            for j in np.flatnonzero(ready[r]):
                self.food_pos[r, j] = spawn_food_point(self.world_rngs[r], arena, self.histories[r],
                                                       self.wcfg, extra=[trunk])
                self.food_active[r, j] = True
                self.food_timer[r, j] = 0

    # --- inspection and state ------------------------------------------------

    def world_state(self, r: int) -> dict[str, Any]:
        """Everything that determines environment ``r``'s future given the same actions."""
        return {
            "pos": self.pos[r].copy(), "heading": float(self.heading[r]), "q": self.q[r].copy(),
            "dq": self.dq[r].copy(), "energy": float(self.energy[r]), "fatigue": float(self.fatigue[r]),
            "food_pos": self.food_pos[r].copy(), "food_active": self.food_active[r].copy(),
            "food_timer": self.food_timer[r].copy(), "neutral_pos": self.neutral_pos[r].copy(),
            "rng": self.world_rngs[r].bit_generator.state, "world_steps": int(self.world_steps[r]),
        }

    def get_state(self) -> dict[str, Any]:
        """Complete state of every environment (world, body, generators, link)."""
        return {
            "n_envs": self.n,
            "history_ids": list(self.history_ids),
            "arrays": {k: getattr(self, k).copy() for k in _STATE_ARRAYS},
            "last_obs": self._last_obs.copy(),
            "seeded": self._seeded.copy(),
            "world_rngs": [None if g is None else json.dumps(g.bit_generator.state) for g in self.world_rngs],
            "link": self.link.get_state(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if int(state["n_envs"]) != self.n or list(state["history_ids"]) != self.history_ids:
            raise ValueError("state belongs to a batch with other environments")
        for k in _STATE_ARRAYS:
            if k == "metabolism" and k not in state["arrays"]:  # states saved before Phase 2b
                self.metabolism[...] = 1.0
                continue
            getattr(self, k)[...] = state["arrays"][k]
        self._last_obs = np.array(state["last_obs"], dtype=float)
        self._seeded[...] = state["seeded"]
        for r, s in enumerate(state["world_rngs"]):
            if s is None:
                self.world_rngs[r] = None
            else:
                g = np.random.default_rng()
                g.bit_generator.state = json.loads(s)
                self.world_rngs[r] = g
        self.link.set_state(state["link"])

    def observation(self) -> dict[str, np.ndarray]:
        """The last observation of every environment (as the mind received it)."""
        return self.layout.split(self._last_obs)


def _clear(p: np.ndarray, r: float, pos: np.ndarray, rad: np.ndarray, clearance: float) -> bool:
    if not len(rad):
        return True
    return bool((np.sqrt(((pos - p) ** 2).sum(1)) >= rad + r + clearance).all())
