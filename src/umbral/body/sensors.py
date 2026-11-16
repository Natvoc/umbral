"""Raw sensors and the fixed observation layout.

Observation (default ``body.vision_rays = 7`` -> 107 floats), channel by channel,
in flat-vector order:

``proprio`` (10) - flat[0:10]
    [0:4]  joint angles ``[left_q1, left_q2, right_q1, right_q2]``, each mapped
           linearly from its joint limits to [-1, 1]
    [4:8]  joint angular velocities actually performed, same order, divided by
           ``max_joint_speed`` -> [-1, 1]
    [8:10] trunk orientation ``[sin(heading), cos(heading)]``

``touch`` (10 x 6 = 60) - flat[10:70], reshaped ``(10, 6)``
    rows 0-7: trunk contour sensors at ``heading + k * 45 deg`` (k = 0 is the
              front, counter-clockwise). Each covers a 45 deg arc centred on it.
    row 8:    left limb tip; row 9: right limb tip.
    columns:  ``[intensity, wall, food, obstacle, neutral, self]``; intensity is
              ``clip(1 - gap / touch_range, 0, 1)`` (1 when touching or
              penetrating); the one-hot names the kind of the strongest contact
              (the first one on ties) and is all zero when intensity is 0.
    Self-contact is felt on both sides: a limb tip touching the trunk activates
    that tip (type self) and the contour sensor whose arc contains the contact
    point (type self). A tip touching the other limb activates the tip.

``vision`` (7 x 5 = 35) - flat[70:105], reshaped ``(7, 5)``
    rays from the trunk's front edge, evenly spread over ``vision_fov`` from
    right (-fov/2) to left (+fov/2). Columns: ``[distance, wall, food, obstacle,
    neutral]``; distance is ``hit / vision_range`` in [0, 1] (1 = nothing within
    range, one-hot all zero). Vision does not see the own body.

``intero`` (2) - flat[105:107]
    ``[energy, fatigue]``, both in [0, 1].

Body parts for the BodyLink:
- ``left_limb`` / ``right_limb``: their 2 joint angles, 2 joint velocities, tip
  touch sensor and 2 motor components.
- ``trunk``: orientation, the 8 contour touch sensors, vision, interoception,
  and the forward/turn motor components.

The ``*_b`` functions are the batched kernels; the others wrap a batch of one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from umbral.body.body import BodyParams, BodyState
from umbral.world.arena import (N_TOUCH_TYPES, N_VISION_TYPES, SELF, WALL, WALL_NORMAL_ANGLES, Arena,
                                cast_rays, norm2, point_segment_distance, unit, wall_gaps, wrap_angle)

CHANNELS = ("proprio", "touch", "vision", "intero")
PARTS = ("trunk", "left_limb", "right_limb")
N_CONTOUR = 8
N_TOUCH = N_CONTOUR + 2
TOUCH_WIDTH = 1 + N_TOUCH_TYPES
VISION_WIDTH = 1 + N_VISION_TYPES


@dataclass(frozen=True)
class Segment:
    channel: str
    part: str
    idx: np.ndarray  # indices into the flat vector


@dataclass(frozen=True)
class ObsLayout:
    n_rays: int

    @property
    def slices(self) -> dict[str, slice]:
        t0 = 10
        v0 = t0 + N_TOUCH * TOUCH_WIDTH
        i0 = v0 + self.n_rays * VISION_WIDTH
        return {"proprio": slice(0, t0), "touch": slice(t0, v0),
                "vision": slice(v0, i0), "intero": slice(i0, i0 + 2)}

    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {"proprio": (10,), "touch": (N_TOUCH, TOUCH_WIDTH),
                "vision": (self.n_rays, VISION_WIDTH), "intero": (2,)}

    @property
    def dim(self) -> int:
        return self.slices["intero"].stop

    def segments(self) -> list[Segment]:
        s = self.slices
        t0, v, i = s["touch"].start, s["vision"], s["intero"]
        rng = lambda a, b: np.arange(a, b)  # noqa: E731
        tip = lambda k: rng(t0 + k * TOUCH_WIDTH, t0 + (k + 1) * TOUCH_WIDTH)  # noqa: E731
        return [
            Segment("proprio", "left_limb", np.array([0, 1, 4, 5])),
            Segment("proprio", "right_limb", np.array([2, 3, 6, 7])),
            Segment("proprio", "trunk", np.array([8, 9])),
            Segment("touch", "trunk", rng(t0, t0 + N_CONTOUR * TOUCH_WIDTH)),
            Segment("touch", "left_limb", tip(N_CONTOUR)),
            Segment("touch", "right_limb", tip(N_CONTOUR + 1)),
            Segment("vision", "trunk", rng(v.start, v.stop)),
            Segment("intero", "trunk", rng(i.start, i.stop)),
        ]

    def split(self, flat: np.ndarray) -> dict[str, np.ndarray]:
        """Channel dict (reshaped copies) from a flat vector (..., dim)."""
        lead = flat.shape[:-1]
        out = {k: flat[..., sl].reshape(*lead, *self.shapes[k]).copy() for k, sl in self.slices.items()}
        out["flat"] = flat.copy()
        return out


MOTOR_SEGMENTS = [
    Segment("motor", "trunk", np.array([0, 1])),
    Segment("motor", "left_limb", np.array([2, 3])),
    Segment("motor", "right_limb", np.array([4, 5])),
]


def _intensity(gap: np.ndarray, touch_range: float) -> np.ndarray:
    return np.clip(1.0 - gap / touch_range, 0.0, 1.0)


# --- batched kernels ------------------------------------------------------------

def touch_b(p: BodyParams, size: float, pos: np.ndarray, heading: np.ndarray, limbs: np.ndarray,
            obj_pos: np.ndarray, obj_r: np.ndarray, obj_kind: np.ndarray, obj_on: np.ndarray) -> np.ndarray:
    """(B, 10, 6) touch array; see module docstring."""
    B = pos.shape[0]
    R, tr, lr = p.trunk_radius, p.touch_range, p.limb_radius
    out = np.zeros((B, N_TOUCH, TOUCH_WIDTH))
    tips = limbs[:, :, 2]                                          # (B, 2, 2)

    # Trunk contour: every contact as (gap, world angle of contact point, kind).
    diff = obj_pos - pos[:, None, :]
    tip_diff = tips - pos[:, None, :]
    gaps = np.concatenate([
        wall_gaps(pos, size) - R,
        norm2(diff) - R - obj_r,
        norm2(tip_diff) - R - lr,
    ], axis=1)                                                     # (B, C)
    angles = np.concatenate([
        np.broadcast_to(WALL_NORMAL_ANGLES, (B, 4)),
        np.arctan2(diff[..., 1], diff[..., 0]),
        np.arctan2(tip_diff[..., 1], tip_diff[..., 0]),
    ], axis=1)
    kinds = np.concatenate([np.full(4, WALL), obj_kind, np.full(2, SELF)])
    on = np.concatenate([np.ones((B, 4), bool), obj_on, np.ones((B, 2), bool)], axis=1)
    inten = np.where(on, _intensity(gaps, tr), 0.0)
    sector = 2 * np.pi / N_CONTOUR
    k = np.rint(wrap_angle(angles - heading[:, None]) / sector).astype(int) % N_CONTOUR
    # Strongest contact per (row, sector); on ties the first contact wins. Only the few
    # active contacts are considered, sorted by (row, sector, -intensity, contact index).
    b, c = np.nonzero(inten > 0)
    if len(b):
        kk, v = k[b, c], inten[b, c]
        order = np.lexsort((c, -v, kk, b))
        b, c, kk, v = b[order], c[order], kk[order], v[order]
        first = np.ones(len(b), bool)
        first[1:] = (b[1:] != b[:-1]) | (kk[1:] != kk[:-1])
        out[b[first], kk[first], 0] = v[first]
        out[b[first], kk[first], 1 + kinds[c[first]]] = 1.0

    # Limb tips: walls, objects, own trunk, the other limb.
    other = limbs[:, ::-1]                                         # (B, 2, 3, 2)
    tip_other = np.minimum(point_segment_distance(tips, other[:, :, 0], other[:, :, 1]),
                           point_segment_distance(tips, other[:, :, 1], other[:, :, 2]))
    d = obj_pos[:, None, :, :] - tips[:, :, None, :]               # (B, 2, M, 2)
    g = np.concatenate([
        wall_gaps(tips, size) - lr,
        norm2(d) - obj_r[:, None, :] - lr,
        (norm2(tip_diff) - R - lr)[..., None],
        (tip_other - 2 * lr)[..., None],
    ], axis=-1)                                                    # (B, 2, 4+M+2)
    tip_on = np.concatenate([np.ones((B, 2, 4), bool), np.broadcast_to(obj_on[:, None, :], d.shape[:-1]),
                             np.ones((B, 2, 2), bool)], axis=-1)
    tip_kinds = np.concatenate([np.full(4, WALL), obj_kind, [SELF, SELF]])
    it = np.where(tip_on, _intensity(g, tr), 0.0)
    j = np.argmax(it, axis=-1)                                     # (B, 2)
    tv = np.take_along_axis(it, j[..., None], axis=-1)[..., 0]
    out[:, N_CONTOUR:, 0] = tv
    rows, limb = np.nonzero(tv > 0)
    out[rows, N_CONTOUR + limb, 1 + tip_kinds[j[rows, limb]]] = 1.0
    return out


def vision_b(p: BodyParams, size: float, pos: np.ndarray, heading: np.ndarray,
             obj_pos: np.ndarray, obj_r: np.ndarray, obj_kind: np.ndarray, obj_on: np.ndarray) -> np.ndarray:
    origin = pos + p.trunk_radius * unit(heading)
    angles = heading[:, None] + np.linspace(-p.vision_fov / 2, p.vision_fov / 2, p.vision_rays)
    dist, kind = cast_rays(origin, angles, size, p.vision_range, obj_pos, obj_r, obj_kind, obj_on)
    out = np.zeros((pos.shape[0], p.vision_rays, VISION_WIDTH))
    out[..., 0] = dist / p.vision_range
    rows, rays = np.nonzero(kind >= 0)
    out[rows, rays, 1 + kind[rows, rays]] = 1.0
    return out


def tip_food_b(p: BodyParams, limbs: np.ndarray, food_pos: np.ndarray, food_r: np.ndarray,
               food_on: np.ndarray) -> np.ndarray:
    """(B, F) bool: active food touched (gap <= 0) by either limb tip."""
    tips = limbs[:, :, 2]
    gap = norm2(food_pos[:, None, :, :] - tips[:, :, None, :]) - food_r[:, None, :] - p.limb_radius
    return (gap <= 0).any(axis=1) & food_on


def raw_observation_b(p: BodyParams, size: float, pos: np.ndarray, heading: np.ndarray, q: np.ndarray,
                      dq: np.ndarray, energy: np.ndarray, fatigue: np.ndarray, limbs: np.ndarray,
                      obj_pos: np.ndarray, obj_r: np.ndarray, obj_kind: np.ndarray, obj_on: np.ndarray
                      ) -> np.ndarray:
    """The body's true signals, (B, dim) float64 (before the BodyLink)."""
    B = pos.shape[0]
    q_norm = 2.0 * (q - p.joint_lo) / (p.joint_hi - p.joint_lo) - 1.0
    return np.concatenate([
        q_norm.reshape(B, 4), (dq / p.max_joint_speed).reshape(B, 4),
        np.sin(heading)[:, None], np.cos(heading)[:, None],
        touch_b(p, size, pos, heading, limbs, obj_pos, obj_r, obj_kind, obj_on).reshape(B, -1),
        vision_b(p, size, pos, heading, obj_pos, obj_r, obj_kind, obj_on).reshape(B, -1),
        energy[:, None], fatigue[:, None],
    ], axis=1)


# --- unbatched wrappers (batch of one) ---------------------------------------------

def _one(objects: tuple[np.ndarray, np.ndarray, np.ndarray]):
    pos, r, kind = objects
    return pos[None], r[None], np.asarray(kind, int), np.ones((1, len(r)), bool)


def touch_sensors(p: BodyParams, arena: Arena, st: BodyState, limbs: np.ndarray,
                  objects: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """(10, 6) touch array for one body; ``objects`` as returned by ``Arena.objects()``."""
    return touch_b(p, arena.size, st.pos[None], np.array([st.heading]), limbs[None], *_one(objects))[0]


def vision_sensors(p: BodyParams, arena: Arena, st: BodyState,
                   objects: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    return vision_b(p, arena.size, st.pos[None], np.array([st.heading]), *_one(objects))[0]


def tip_food_contacts(p: BodyParams, arena: Arena, limbs: np.ndarray) -> np.ndarray:
    """Indices of active food items touched (gap <= 0) by either limb tip."""
    touched = tip_food_b(p, limbs[None], arena.food_pos[None], arena.food_r[None], arena.food_active[None])
    return np.flatnonzero(touched[0])


def raw_observation(p: BodyParams, arena: Arena, st: BodyState, limbs: np.ndarray) -> np.ndarray:
    return raw_observation_b(p, arena.size, st.pos[None], np.array([st.heading]), st.q[None], st.dq[None],
                             np.array([st.energy]), np.array([st.fatigue]), limbs[None],
                             *_one(arena.objects()))[0]


__all__ = ["CHANNELS", "PARTS", "MOTOR_SEGMENTS", "ObsLayout", "Segment", "raw_observation", "raw_observation_b",
           "tip_food_contacts", "tip_food_b", "touch_b", "touch_sensors", "vision_b", "vision_sensors"]
