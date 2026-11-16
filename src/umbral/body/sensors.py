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
              and is all zero when intensity is 0.
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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from umbral.body.body import BodyParams, BodyState, wrap_angle
from umbral.world.arena import (N_TOUCH_TYPES, N_VISION_TYPES, SELF, WALL, WALL_NORMAL_ANGLES, Arena,
                                cast_rays, point_segment_distance, unit, wall_gaps)

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
        """Channel dict (reshaped copies) from a flat vector."""
        out = {k: flat[sl].reshape(self.shapes[k]).copy() for k, sl in self.slices.items()}
        out["flat"] = flat.copy()
        return out


MOTOR_SEGMENTS = [
    Segment("motor", "trunk", np.array([0, 1])),
    Segment("motor", "left_limb", np.array([2, 3])),
    Segment("motor", "right_limb", np.array([4, 5])),
]


def _intensity(gap: np.ndarray, touch_range: float) -> np.ndarray:
    return np.clip(1.0 - gap / touch_range, 0.0, 1.0)


def touch_sensors(p: BodyParams, arena: Arena, st: BodyState, limbs: np.ndarray,
                  objects: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """(10, 6) touch array; see module docstring."""
    out = np.zeros((N_TOUCH, TOUCH_WIDTH))
    obj_pos, obj_r, obj_kind = objects
    c, R, tr, lr = st.pos, p.trunk_radius, p.touch_range, p.limb_radius
    tips = limbs[:, 2]

    # Trunk contour: every contact as (gap, world angle of contact point, kind).
    diff = obj_pos - c
    tip_diff = tips - c
    gaps = np.concatenate([
        wall_gaps(c[None], arena.size)[0] - R,
        np.sqrt((diff * diff).sum(1)) - R - obj_r,
        np.sqrt((tip_diff * tip_diff).sum(1)) - R - lr,
    ])
    angles = np.concatenate([WALL_NORMAL_ANGLES, np.arctan2(diff[:, 1], diff[:, 0]),
                             np.arctan2(tip_diff[:, 1], tip_diff[:, 0])])
    kinds = np.concatenate([np.full(4, WALL), obj_kind, np.full(2, SELF)])
    inten = _intensity(gaps, tr)
    sector = 2 * np.pi / N_CONTOUR
    for j in np.flatnonzero(inten > 0):
        k = int(round(wrap_angle(angles[j] - st.heading) / sector)) % N_CONTOUR
        if inten[j] > out[k, 0]:
            out[k] = 0.0
            out[k, 0] = inten[j]
            out[k, 1 + kinds[j]] = 1.0

    # Limb tips.
    for i in range(2):
        t = tips[i]
        d = obj_pos - t
        other = limbs[1 - i]
        g = np.concatenate([
            wall_gaps(t[None], arena.size)[0] - lr,
            np.sqrt((d * d).sum(1)) - obj_r - lr,
            [np.linalg.norm(t - c) - R - lr,
             min(point_segment_distance(t, other[0], other[1]),
                 point_segment_distance(t, other[1], other[2])) - 2 * lr],
        ])
        kk = np.concatenate([np.full(4, WALL), obj_kind, [SELF, SELF]])
        it = _intensity(g, tr)
        j = int(np.argmax(it))
        if it[j] > 0:
            out[N_CONTOUR + i, 0] = it[j]
            out[N_CONTOUR + i, 1 + kk[j]] = 1.0
    return out


def tip_food_contacts(p: BodyParams, arena: Arena, limbs: np.ndarray) -> np.ndarray:
    """Indices of active food items touched (gap <= 0) by either limb tip."""
    if not len(arena.food_r):
        return np.zeros(0, int)
    tips = limbs[:, 2]
    d = np.sqrt(((arena.food_pos[None] - tips[:, None]) ** 2).sum(-1))  # (2, F)
    touched = (d - arena.food_r[None] - p.limb_radius <= 0).any(0) & arena.food_active
    return np.flatnonzero(touched)


def vision_sensors(p: BodyParams, arena: Arena, st: BodyState,
                   objects: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    origin = st.pos + p.trunk_radius * unit(st.heading)
    angles = st.heading + np.linspace(-p.vision_fov / 2, p.vision_fov / 2, p.vision_rays)
    dist, kind = cast_rays(origin, angles, arena.size, p.vision_range, *objects)
    out = np.zeros((p.vision_rays, VISION_WIDTH))
    out[:, 0] = dist / p.vision_range
    hit = kind >= 0
    out[np.flatnonzero(hit), 1 + kind[hit]] = 1.0
    return out


def raw_observation(p: BodyParams, arena: Arena, st: BodyState, limbs: np.ndarray) -> np.ndarray:
    """The body's true signals as a flat float64 vector (before the BodyLink)."""
    objects = arena.objects()
    q_norm = 2.0 * (st.q - p.joint_lo) / (p.joint_hi - p.joint_lo) - 1.0
    proprio = np.concatenate([q_norm.ravel(), (st.dq / p.max_joint_speed).ravel(),
                              [np.sin(st.heading), np.cos(st.heading)]])
    return np.concatenate([
        proprio,
        touch_sensors(p, arena, st, limbs, objects).ravel(),
        vision_sensors(p, arena, st, objects).ravel(),
        [st.energy, st.fatigue],
    ])


__all__ = ["CHANNELS", "PARTS", "MOTOR_SEGMENTS", "ObsLayout", "Segment", "raw_observation",
           "tip_food_contacts", "touch_sensors", "vision_sensors"]
