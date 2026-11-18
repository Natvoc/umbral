"""Behaviour battery (Phase 2b, pre-specified before the pilot): the seed of the Phase 4 battery.

An agent (mind + policy, no learning) lives ``behavior.episodes`` episodes of
``behavior.steps`` steps in its history, on evaluation seeds
(``BEHAVIOR_SEED_BASE + i``, disjoint from training and from the Phase 2a
evaluation). The policy keeps its noise, drawn from per-episode generators: style
includes how variable the agent is. Measures per episode (experimenter side, from
the true world state):

- ``distance``: path length of the trunk centre;
- ``area``: fraction of the cells of a ``behavior.grid`` grid over the arena visited
  by the trunk centre;
- ``energy_mean``, ``energy_below_0.3``, ``energy_below_0.1``: mean energy and the
  fractions of steps below 0.3 and 0.1;
- ``food``: items eaten;
- ``effort``: mean ``mean(executed**2)`` per step;
- ``neutral_contacts``: contacts with neutral objects, counted as onsets (steps where
  some sensor touches a neutral object and none did the step before). Neutral
  objects give nothing: this is exploration for its own sake;
- reaction to the unexpected: every ``probe_every`` steps (first at
  ``probe_offset``) a neutral object appears ``probe_distance`` ahead, inside the
  visual field (the first free direction among 0, +-0.3, +-0.6 rad from the heading;
  skipped if none is free). It is watched for ``probe_window`` steps, then removed.
  ``probe_delta``: mean change of the trunk-probe distance over the window (negative
  = approached). ``probe_latency``: mean number of steps until that distance first
  changed by more than ``probe_threshold`` (the window length if it never did).
  ``probe_approach``: fraction of probes whose first such change was an approach
  (only probes with a change).

Comparisons between agents (``compare``): per measure, the mean of each agent with
a 95 % bootstrap CI over episodes, and Hedges' g for every pair of agents, with a
bootstrap CI (resampling episodes within each agent). With one agent per
temperament, this is variability *within* an individual, not between individuals.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import torch

from umbral.config import Config
from umbral.mind.agent import Agent, GaussianNoise
from umbral.mind.training import EVAL_SEED_BASE, POLICY_SEED_OFFSET
from umbral.world.arena import NEUTRAL
from umbral.world.batched import BatchedUmbralEnv

BEHAVIOR_SEED_BASE = EVAL_SEED_BASE + 3_000_000
MEASURES = ("distance", "area", "energy_mean", "energy_below_0.3", "energy_below_0.1", "food", "effort",
            "neutral_contacts", "probe_delta", "probe_latency", "probe_approach")
PROBE_ANGLES = (0.0, 0.3, -0.3, 0.6, -0.6)


def _probe_spot(env: BatchedUmbralEnv, r: int, distance: float, radius: float) -> np.ndarray | None:
    size = env.size
    for da in PROBE_ANGLES:
        ang = env.heading[r] + da
        p = env.pos[r] + distance * np.array([np.cos(ang), np.sin(ang)])
        if not ((p >= radius + 0.1).all() and (p <= size - radius - 0.1).all()):
            continue
        blockers = [(env.obstacle_pos[r][env.obstacle_on[r]], env.obstacle_r[r][env.obstacle_on[r]]),
                    (env.neutral_pos[r][env.neutral_on[r]], env.neutral_r[r][env.neutral_on[r]]),
                    (env.food_pos[r][env.food_active[r]], env.food_r[r][env.food_active[r]])]
        if all((np.sqrt(((bp - p) ** 2).sum(-1)) >= br + radius + 0.1).all() for bp, br in blockers):
            return p
    return None


@torch.no_grad()
def run_battery(agent: Agent, config: Config, history: str, episodes: int | None = None,
                steps: int | None = None) -> dict[str, list[float]]:
    """The battery for one agent; returns ``{measure: [value per episode]}``."""
    bc = config.behavior
    n = int(bc.episodes if episodes is None else episodes)
    T = int(bc.steps if steps is None else steps)
    seeds = [BEHAVIOR_SEED_BASE + i for i in range(n)]
    env = BatchedUmbralEnv(config, n, [str(history)] * n)
    env.reset(seed=seeds)
    noise = GaussianNoise(n, [s + POLICY_SEED_OFFSET for s in seeds])
    world = agent.world
    h = world.initial_state(n)
    prev = np.zeros((n, 6))
    o = env.observation()["flat"]
    G = int(round(env.size / float(bc.grid)))
    visited = np.zeros((n, G, G), bool)
    radius = float(config.world.neutral_radius)
    every, offset, window = int(bc.probe_every), int(bc.probe_offset), int(bc.probe_window)
    thr, dist = float(bc.probe_threshold), float(bc.probe_distance)

    def mark() -> None:
        c = np.clip((env.pos / float(bc.grid)).astype(int), 0, G - 1)
        visited[np.arange(n), c[:, 0], c[:, 1]] = True

    acc = {k: np.zeros(n) for k in ("distance", "energy", "below03", "below01", "food", "effort", "neutral")}
    probes: list[list[tuple[float, int, int]]] = [[] for _ in range(n)]  # (delta, latency, sign)
    active: dict[int, dict[str, Any]] = {}
    was_neutral = np.zeros(n, bool)
    tl = env.layout.slices["touch"]
    mark()
    for t in range(T):
        if t >= offset and (t - offset) % every == 0:
            for r in range(n):
                spot = _probe_spot(env, r, dist, radius)
                if spot is not None:
                    env.place_probe(r, spot, radius)
                    d0 = float(np.linalg.norm(env.pos[r] - spot))
                    active[r] = {"t0": t, "d0": d0, "first": None}
        h = world.step_hidden(torch.as_tensor(o, dtype=torch.float32), torch.as_tensor(prev, dtype=torch.float32), h)
        a, *_ = agent.act(h[0], noise.sample())
        p0 = env.pos.copy()
        obs, _, _, _, info = env.step(a)
        o, prev = obs["flat"], info["action_intended"]
        acc["distance"] += np.sqrt(((env.pos - p0) ** 2).sum(-1))
        mark()
        e = info["energy"]
        acc["energy"] += e
        acc["below03"] += e < 0.3
        acc["below01"] += e < 0.1
        acc["food"] += info["food_eaten"]
        acc["effort"] += (info["action_executed"] ** 2).mean(-1)
        touch = env.raw[:, tl].reshape(n, 10, 6)
        now_neutral = ((touch[..., 0] > 0) & (touch[..., 1 + NEUTRAL] == 1)).any(-1)
        acc["neutral"] += now_neutral & ~was_neutral
        was_neutral = now_neutral
        for r in list(active):
            pr = active[r]
            d = float(np.linalg.norm(env.pos[r] - env.neutral_pos[r, env.probe_slot]))
            k = t - pr["t0"] + 1
            if pr["first"] is None and abs(d - pr["d0"]) > thr:
                pr["first"] = (k, -1 if d < pr["d0"] else 1)
            if k >= window:
                lat, sign = pr["first"] if pr["first"] is not None else (window, 0)
                probes[r].append((d - pr["d0"], lat, sign))
                env.clear_probe(r)
                del active[r]
    out = {
        "distance": acc["distance"].tolist(),
        "area": (visited.reshape(n, -1).mean(-1)).tolist(),
        "energy_mean": (acc["energy"] / T).tolist(),
        "energy_below_0.3": (acc["below03"] / T).tolist(),
        "energy_below_0.1": (acc["below01"] / T).tolist(),
        "food": acc["food"].tolist(),
        "effort": (acc["effort"] / T).tolist(),
        "neutral_contacts": acc["neutral"].tolist(),
        "probe_delta": [float(np.mean([p[0] for p in ps])) if ps else float("nan") for ps in probes],
        "probe_latency": [float(np.mean([p[1] for p in ps])) if ps else float("nan") for ps in probes],
        "probe_approach": [float(np.mean([p[2] < 0 for p in ps if p[2] != 0])) if any(p[2] for p in ps)
                           else float("nan") for ps in probes],
        "probes": [len(ps) for ps in probes],
    }
    return out


def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    n1, n2 = len(x), len(y)
    sp = np.sqrt(((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2))
    if sp == 0:
        return 0.0 if x.mean() == y.mean() else float(np.sign(x.mean() - y.mean()) * np.inf)
    return float((x.mean() - y.mean()) / sp * (1 - 3 / (4 * (n1 + n2) - 9)))


def compare(results: dict[str, dict[str, list[float]]], n_boot: int, seed: int = 0) -> dict[str, Any]:
    """Means with CIs per agent, and Hedges' g with CIs for every pair, per measure."""
    rng = np.random.default_rng(seed)
    names = list(results)
    out: dict[str, Any] = {"means": {}, "hedges_g": {}, "n_boot": n_boot}
    for m in MEASURES:
        out["means"][m] = {}
        vals = {a: np.asarray(results[a][m], float) for a in names}
        vals = {a: v[~np.isnan(v)] for a, v in vals.items()}
        for a, v in vals.items():
            if len(v) == 0:
                out["means"][m][a] = None
                continue
            bs = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
            out["means"][m][a] = {"mean": float(v.mean()), "ci95": [float(np.quantile(bs, 0.025)),
                                                                     float(np.quantile(bs, 0.975))]}
        out["hedges_g"][m] = {}
        for a, b in itertools.combinations(names, 2):
            x, y = vals[a], vals[b]
            if len(x) < 2 or len(y) < 2:
                out["hedges_g"][m][f"{a} vs {b}"] = None
                continue
            g = hedges_g(x, y)
            bs = np.array([hedges_g(x[rng.integers(0, len(x), len(x))], y[rng.integers(0, len(y), len(y))])
                           for _ in range(n_boot)])
            bs = bs[np.isfinite(bs)]
            ci = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))] if len(bs) else [None, None]
            out["hedges_g"][m][f"{a} vs {b}"] = {"g": g, "ci95": ci}
    return out
