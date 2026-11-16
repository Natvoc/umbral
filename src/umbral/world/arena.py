"""Arena state and geometry: a closed square with circular objects.

Coordinates: ``x`` and ``y`` in ``[0, size]``; angles in radians, counter-clockwise
from +x. All objects are circles. Object kinds share one integer code space with
the touch/vision type one-hots:

    WALL=0, FOOD=1, OBSTACLE=2, NEUTRAL=3, SELF=4  (SELF is touch-only)

Only the numpy ``Generator`` passed in is used for randomness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WALL, FOOD, OBSTACLE, NEUTRAL, SELF = 0, 1, 2, 3, 4
KIND_NAMES = ("wall", "food", "obstacle", "neutral", "self")
N_TOUCH_TYPES = 5   # wall, food, obstacle, neutral, self
N_VISION_TYPES = 4  # wall, food, obstacle, neutral (vision does not see the own body)


@dataclass
class Arena:
    size: float
    obstacle_pos: np.ndarray   # (O, 2), fixed
    obstacle_r: np.ndarray     # (O,)
    neutral_pos: np.ndarray    # (N, 2), pushable
    neutral_r: np.ndarray      # (N,)
    food_pos: np.ndarray       # (F, 2)
    food_r: np.ndarray         # (F,)
    food_active: np.ndarray    # (F,) bool
    food_timer: np.ndarray     # (F,) int, steps until an inactive food reappears

    def copy(self) -> "Arena":
        return Arena(self.size, *(np.array(getattr(self, f)) for f in (
            "obstacle_pos", "obstacle_r", "neutral_pos", "neutral_r",
            "food_pos", "food_r", "food_active", "food_timer")))

    def objects(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Active objects as (pos (M,2), radius (M,), kind (M,)); inactive food excluded."""
        act = self.food_active
        pos = np.concatenate([self.food_pos[act], self.obstacle_pos, self.neutral_pos])
        r = np.concatenate([self.food_r[act], self.obstacle_r, self.neutral_r])
        kind = np.concatenate([
            np.full(int(act.sum()), FOOD), np.full(len(self.obstacle_r), OBSTACLE),
            np.full(len(self.neutral_r), NEUTRAL),
        ])
        return pos, r, kind


# --- geometry -----------------------------------------------------------------

def unit(angle: np.ndarray | float) -> np.ndarray:
    angle = np.asarray(angle, dtype=float)
    return np.stack([np.cos(angle), np.sin(angle)], axis=-1)


def wall_gaps(points: np.ndarray, size: float) -> np.ndarray:
    """Distance from each point (P,2) to the 4 walls: (P,4) = [left, right, bottom, top]."""
    x, y = points[:, 0], points[:, 1]
    return np.stack([x, size - x, y, size - y], axis=1)


# Outward normal angle of each wall, in the order used by ``wall_gaps``.
WALL_NORMAL_ANGLES = np.array([np.pi, 0.0, -np.pi / 2, np.pi / 2])


def cast_rays(origin: np.ndarray, angles: np.ndarray, size: float, max_range: float,
              obj_pos: np.ndarray, obj_r: np.ndarray, obj_kind: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray]:
    """First hit along each ray from ``origin``.

    Returns (distance (R,), kind (R,)); distance is ``max_range`` and kind -1 when
    nothing is hit within range.
    """
    d = unit(angles)                                         # (R,2)
    # Walls: the ray always leaves the arena through some wall.
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(d[:, 0] > 0, (size - origin[0]) / d[:, 0],
                      np.where(d[:, 0] < 0, -origin[0] / d[:, 0], np.inf))
        ty = np.where(d[:, 1] > 0, (size - origin[1]) / d[:, 1],
                      np.where(d[:, 1] < 0, -origin[1] / d[:, 1], np.inf))
    t_best = np.minimum(tx, ty)
    kind_best = np.full(len(angles), WALL)

    if len(obj_r):
        oc = obj_pos - origin                               # (M,2)
        t_ca = d @ oc.T                                     # (R,M)
        d2 = (oc * oc).sum(1)[None, :] - t_ca**2
        disc = obj_r[None, :] ** 2 - d2
        hit = disc >= 0
        sq = np.sqrt(np.where(hit, disc, 0.0))
        t_hit = t_ca - sq
        inside = (oc * oc).sum(1) <= obj_r**2               # origin inside the object
        t_hit = np.where(inside[None, :], 0.0, t_hit)
        valid = hit & (t_hit >= 0)
        t_hit = np.where(valid, t_hit, np.inf)
        j = np.argmin(t_hit, axis=1)
        t_obj = t_hit[np.arange(len(angles)), j]
        closer = t_obj < t_best
        t_best = np.where(closer, t_obj, t_best)
        kind_best = np.where(closer, obj_kind[j], kind_best)

    miss = t_best >= max_range
    return np.where(miss, max_range, t_best), np.where(miss, -1, kind_best)


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom == 0 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def push_out_of_circles(center: np.ndarray, radius: float, pos: np.ndarray, r: np.ndarray,
                        iterations: int = 3) -> np.ndarray:
    """Move a circle so it does not overlap any of the given fixed circles."""
    c = center.astype(float).copy()
    if not len(r):
        return c
    for _ in range(iterations):
        diff = c - pos
        dist = np.sqrt((diff * diff).sum(1))
        overlap = radius + r - dist
        mask = overlap > 0
        if not mask.any():
            break
        safe = np.where(dist[mask] > 1e-9, dist[mask], 1.0)
        normals = diff[mask] / safe[:, None]
        normals[dist[mask] <= 1e-9] = (1.0, 0.0)
        c += (normals * overlap[mask][:, None]).sum(0)
    return c


def clamp_in_arena(center: np.ndarray, radius: float, size: float) -> np.ndarray:
    return np.clip(center, radius, size - radius)
