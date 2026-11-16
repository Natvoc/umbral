"""Arena state and batched geometry: closed squares with circular objects.

Coordinates: ``x`` and ``y`` in ``[0, size]``; angles in radians, counter-clockwise
from +x. All objects are circles. Object kinds share one integer code space with
the touch/vision type one-hots:

    WALL=0, FOOD=1, OBSTACLE=2, NEUTRAL=3, SELF=4  (SELF is touch-only)

Batching rule (exact equivalence between a batch and single environments): every
function here works on arrays with a leading batch axis and computes each row
with the same elementwise operations, whatever the batch size. Reductions run
only along the last axis (whose length does not depend on the batch), and there
is no matrix multiplication (BLAS blocking can depend on the array shape).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WALL, FOOD, OBSTACLE, NEUTRAL, SELF = 0, 1, 2, 3, 4
KIND_NAMES = ("wall", "food", "obstacle", "neutral", "self")
N_TOUCH_TYPES = 5   # wall, food, obstacle, neutral, self
N_VISION_TYPES = 4  # wall, food, obstacle, neutral (vision does not see the own body)
FAR = 1.0e6         # position of padding objects (radius 0): they never interact

# Outward normal angle of each wall, in the order used by ``wall_gaps``.
WALL_NORMAL_ANGLES = np.array([np.pi, 0.0, -np.pi / 2, np.pi / 2])


@dataclass
class Arena:
    """One arena (unbatched). Used to build histories and for hand-built scenes."""
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


# --- elementary geometry (any leading shape) ----------------------------------

def unit(angle: np.ndarray | float) -> np.ndarray:
    angle = np.asarray(angle, dtype=float)
    return np.stack([np.cos(angle), np.sin(angle)], axis=-1)


def norm2(v: np.ndarray) -> np.ndarray:
    """Euclidean norm over the last axis of length 2 (explicit, shape-independent)."""
    return np.sqrt(v[..., 0] * v[..., 0] + v[..., 1] * v[..., 1])


def wrap_angle(a: np.ndarray | float) -> np.ndarray:
    return (np.asarray(a, dtype=float) + np.pi) % (2 * np.pi) - np.pi


def wall_gaps(points: np.ndarray, size: float) -> np.ndarray:
    """Distance from points (..., 2) to the 4 walls -> (..., 4) = [left, right, bottom, top]."""
    x, y = points[..., 0], points[..., 1]
    return np.stack([x, size - x, y, size - y], axis=-1)


def clamp_in_arena(center: np.ndarray, radius: np.ndarray | float, size: float) -> np.ndarray:
    r = np.asarray(radius, dtype=float)
    if r.ndim:
        r = r[..., None]
    return np.clip(center, r, size - r)


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from points p to segments [a, b]; all (..., 2), broadcast together."""
    ab = b - a
    ap = p - a
    denom = ab[..., 0] * ab[..., 0] + ab[..., 1] * ab[..., 1]
    num = ap[..., 0] * ab[..., 0] + ap[..., 1] * ab[..., 1]
    safe = np.where(denom == 0, 1.0, denom)
    t = np.where(denom == 0, 0.0, np.clip(num / safe, 0.0, 1.0))
    closest = a + t[..., None] * ab
    return norm2(p - closest)


def push_out_of_circles(center: np.ndarray, radius: np.ndarray, pos: np.ndarray, r: np.ndarray,
                        iterations: int = 3) -> np.ndarray:
    """Move circles out of fixed circles.

    center (B, K, 2), radius (B, K) or scalar; pos (B, M, 2), r (B, M).
    Overlaps are resolved by summing push vectors over the M blockers (last-axis
    sums, padding contributes exact zeros).
    """
    c = np.array(center, dtype=float, copy=True)
    if pos.shape[-2] == 0:
        return c
    rad = np.broadcast_to(np.asarray(radius, dtype=float), c.shape[:-1])
    for _ in range(iterations):
        dx = c[..., :, None, 0] - pos[..., None, :, 0]          # (B, K, M)
        dy = c[..., :, None, 1] - pos[..., None, :, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = rad[..., :, None] + r[..., None, :] - dist
        hit = overlap > 0
        if not hit.any():
            break  # nothing overlaps anywhere: further iterations would only add exact zeros
        tiny = dist <= 1e-9
        safe = np.where(tiny, 1.0, dist)
        nx = np.where(tiny, 1.0, dx / safe)
        ny = np.where(tiny, 0.0, dy / safe)
        amount = np.where(hit, overlap, 0.0)
        c[..., 0] += np.sum(nx * amount, axis=-1)
        c[..., 1] += np.sum(ny * amount, axis=-1)
    return c


def cast_rays(origin: np.ndarray, angles: np.ndarray, size: float, max_range: float,
              obj_pos: np.ndarray, obj_r: np.ndarray, obj_kind: np.ndarray, obj_on: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray]:
    """First hit along each ray.

    origin (B, 2), angles (B, R), obj_pos (B, M, 2), obj_r (B, M), obj_kind (M,),
    obj_on (B, M) bool. Returns (distance (B, R), kind (B, R)); distance is
    ``max_range`` and kind -1 when nothing is hit within range.
    """
    dxr, dyr = np.cos(angles), np.sin(angles)                     # (B, R)
    ox, oy = origin[:, 0:1], origin[:, 1:2]
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(dxr > 0, (size - ox) / dxr, np.where(dxr < 0, -ox / dxr, np.inf))
        ty = np.where(dyr > 0, (size - oy) / dyr, np.where(dyr < 0, -oy / dyr, np.inf))
    t_best = np.minimum(tx, ty)
    kind_best = np.full(angles.shape, WALL)

    if obj_pos.shape[1]:
        ocx = obj_pos[:, None, :, 0] - origin[:, None, None, 0]   # (B, 1, M)
        ocy = obj_pos[:, None, :, 1] - origin[:, None, None, 1]
        t_ca = dxr[:, :, None] * ocx + dyr[:, :, None] * ocy       # (B, R, M)
        occ = ocx * ocx + ocy * ocy
        r2 = (obj_r * obj_r)[:, None, :]
        d2 = occ - t_ca * t_ca
        disc = r2 - d2
        hit = disc >= 0
        sq = np.sqrt(np.where(hit, disc, 0.0))
        t_hit = t_ca - sq
        t_hit = np.where(occ <= r2, 0.0, t_hit)                   # origin inside the object
        valid = hit & (t_hit >= 0) & obj_on[:, None, :]
        t_hit = np.where(valid, t_hit, np.inf)
        j = np.argmin(t_hit, axis=-1)                              # (B, R)
        t_obj = np.take_along_axis(t_hit, j[..., None], axis=-1)[..., 0]
        closer = t_obj < t_best
        t_best = np.where(closer, t_obj, t_best)
        kind_best = np.where(closer, obj_kind[j], kind_best)

    miss = t_best >= max_range
    return np.where(miss, max_range, t_best), np.where(miss, -1, kind_best)
