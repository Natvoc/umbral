"""Histories: fixed, reproducible versions of the world, defined in ``world.histories``.

A history's ``layout_seed`` fixes its obstacles, neutral objects, initial food and
food patches, so the same history is the same world for every agent. The run
seed only affects the body's start pose and where food reappears.

Obstacle layouts:
- ``scattered``: ``obstacle_count`` circles placed uniformly.
- ``corridors``: two horizontal walls of touching circles, each with a gap at
  opposite ends, plus ``obstacle_count`` scattered circles.
- ``clustered``: ``obstacle_count`` circles grouped around 2 cluster centres.

Food distributions:
- ``uniform``: anywhere free.
- ``patches``: around ``food_patches`` centres (Gaussian, sd = ``patch_radius``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from umbral.config import Config
from umbral.world.arena import Arena

CORRIDOR_RADIUS = 0.3
CORRIDOR_GAP = 3.0
EDGE_MARGIN = 0.3


@dataclass
class History:
    id: str
    name: str
    cfg: Config
    patch_centers: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))


def get_history(world_cfg: Config, history_id: str | int | None = None) -> History:
    hid = str(world_cfg.history_id if history_id is None else history_id)
    try:
        h = world_cfg.histories[hid]
    except KeyError:
        raise KeyError(f"unknown world.history_id {hid!r}; defined: {sorted(world_cfg.histories)}") from None
    return History(id=hid, name=h.name, cfg=h)


def _free(p: np.ndarray, r: float, pos: np.ndarray, rad: np.ndarray, clearance: float) -> bool:
    if not len(rad):
        return True
    d = np.sqrt(((pos - p) ** 2).sum(1))
    return bool((d >= rad + r + clearance).all())


def sample_free_point(rng: np.random.Generator, size: float, r: float, blockers: list[tuple[np.ndarray, np.ndarray]],
                      clearance: float, tries: int, center: np.ndarray | None = None,
                      spread: float = 0.0) -> np.ndarray:
    """Rejection-sample a point whose circle of radius ``r`` avoids every blocker.

    With ``center``, samples around it (Gaussian, sd=``spread``). If every try
    fails, returns the last candidate (overlap is tolerated rather than looping).
    """
    lo, hi = r + EDGE_MARGIN, size - r - EDGE_MARGIN
    p = None
    for _ in range(max(1, tries)):
        if center is None:
            p = rng.uniform(lo, hi, size=2)
        else:
            p = np.clip(center + rng.normal(0.0, spread, size=2), lo, hi)
        if all(_free(p, r, bp, br, clearance) for bp, br in blockers):
            return p
    return p


def _corridor_walls(size: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    step = 2 * CORRIDOR_RADIUS
    xs = np.arange(CORRIDOR_RADIUS, size - CORRIDOR_RADIUS + 1e-9, step)
    left_first = bool(rng.integers(2))
    pos = []
    for row, y in enumerate((size / 3, 2 * size / 3)):
        gap_left = left_first == (row == 0)
        keep = xs > CORRIDOR_GAP if gap_left else xs < size - CORRIDOR_GAP
        pos.extend((x, y) for x in xs[keep])
    pos = np.array(pos, dtype=float)
    return pos, np.full(len(pos), CORRIDOR_RADIUS)


def build_arena(world_cfg: Config, history_id: str | int | None = None) -> tuple[Arena, History]:
    """Deterministic initial arena for a history (default: ``world.history_id``)."""
    hist = get_history(world_cfg, history_id)
    h = hist.cfg
    size = float(world_cfg.size)
    tries = int(world_cfg.spawn_tries)
    rng = np.random.default_rng(int(h.layout_seed))
    rmin, rmax = (float(v) for v in h.obstacle_radius)

    obs_pos = np.zeros((0, 2))
    obs_r = np.zeros(0)
    if h.obstacle_layout == "corridors":
        obs_pos, obs_r = _corridor_walls(size, rng)
    centers = None
    if h.obstacle_layout == "clustered":
        centers = [sample_free_point(rng, size, 1.5, [], 0.0, tries) for _ in range(2)]
    elif h.obstacle_layout not in ("scattered", "corridors"):
        raise ValueError(f"unknown obstacle_layout {h.obstacle_layout!r}")
    for i in range(int(h.obstacle_count)):
        r = float(rng.uniform(rmin, rmax))
        c = None if centers is None else centers[i % len(centers)]
        p = sample_free_point(rng, size, r, [(obs_pos, obs_r)], 0.4, tries, center=c, spread=0.9)
        obs_pos = np.vstack([obs_pos, p])
        obs_r = np.append(obs_r, r)

    if h.food_distribution == "patches":
        n = int(h.food_patches)
        pc = np.zeros((0, 2))
        for _ in range(n):
            p = sample_free_point(rng, size, float(h.patch_radius), [(obs_pos, obs_r), (pc, np.full(len(pc), 2.0))],
                                  0.5, tries)
            pc = np.vstack([pc, p])
        hist.patch_centers = pc
    elif h.food_distribution != "uniform":
        raise ValueError(f"unknown food_distribution {h.food_distribution!r}")

    nr = float(world_cfg.neutral_radius)
    neu_pos = np.zeros((0, 2))
    for _ in range(int(h.neutral_count)):
        p = sample_free_point(rng, size, nr, [(obs_pos, obs_r), (neu_pos, np.full(len(neu_pos), nr))], 0.3, tries)
        neu_pos = np.vstack([neu_pos, p])

    fr = float(world_cfg.food_radius)
    arena = Arena(
        size=size, obstacle_pos=obs_pos, obstacle_r=obs_r,
        neutral_pos=neu_pos, neutral_r=np.full(len(neu_pos), nr),
        food_pos=np.zeros((0, 2)), food_r=np.zeros(0),
        food_active=np.zeros(0, bool), food_timer=np.zeros(0, int),
    )
    food: list[np.ndarray] = []
    for _ in range(int(h.food_count)):
        placed = (np.array(food, dtype=float).reshape(-1, 2), np.full(len(food), fr))
        food.append(spawn_food_point(rng, arena, hist, world_cfg, extra=[placed]))
    arena.food_pos = np.array(food, dtype=float).reshape(-1, 2)
    arena.food_r = np.full(len(food), fr)
    arena.food_active = np.ones(len(food), bool)
    arena.food_timer = np.zeros(len(food), int)
    return arena, hist


def spawn_food_point(rng: np.random.Generator, arena: Arena, hist: History, world_cfg: Config,
                     extra: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Where a food item appears (initially or on respawn), per the history's distribution."""
    fr = float(world_cfg.food_radius)
    act = arena.food_active
    blockers = [(arena.obstacle_pos, arena.obstacle_r), (arena.neutral_pos, arena.neutral_r),
                (arena.food_pos[act], arena.food_r[act]), *extra]
    center = None
    spread = 0.0
    if len(hist.patch_centers):
        center = hist.patch_centers[int(rng.integers(len(hist.patch_centers)))]
        spread = float(hist.cfg.patch_radius)
    return sample_free_point(rng, arena.size, fr, blockers, 0.1, int(world_cfg.spawn_tries),
                             center=center, spread=spread)
