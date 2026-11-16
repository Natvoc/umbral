"""The physical body: trunk, two 2-segment limbs, kinematics, energy and fatigue.

Pure kinematics in numpy, no physics engine.

Kinematics of limb ``i`` (0 = left, 1 = right), with trunk centre ``c``, heading
``h``, trunk radius ``R``, mount angle ``m_i`` and joint angles ``q_i1, q_i2``::

    base  = c + R * u(h + m_i)
    elbow = base + L1 * u(h + m_i + q_i1)
    tip   = elbow + L2 * u(h + m_i + q_i1 + q_i2)

where ``u(a) = (cos a, sin a)``. Both limbs use the same counter-clockwise angle
convention (they are not mirrored).

Action vector (6 values in [-1, 1]):
``[forward, turn, left_q1, left_q2, right_q1, right_q2]``; joint entries are
angular velocities. The *executed* action is the applied action times the
body's current strength (lower at low energy and high fatigue).

Rules:
- The trunk never passes through walls or obstacles (if push-out cannot clear an
  overlap, e.g. wedged between two obstacles, the trunk move is reverted). Neutral objects absorb
  ``push_fraction`` of an overlap (a light push) and never pass through walls or
  obstacles themselves.
- Limbs only sense obstacles, but may not cross walls: a joint move that would put
  a limb point outside the arena is reverted; if the limb is outside even with its
  previous angles (trunk moved too close to a wall), the trunk move is reverted.
- Energy: ``E' = clip(E - basal - effort_cost * effort + food_value * n_eaten, 0, 1)``
  with ``effort = mean(executed ** 2)``. Energy 0 means weaker, slower movement,
  never death.
- Fatigue (provisional): ``F' = clip(decay * F + gain * effort, 0, 1)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from umbral.config import Config
from umbral.world.arena import Arena, clamp_in_arena, push_out_of_circles, unit

ACTION_DIM = 6


@dataclass(frozen=True)
class BodyParams:
    trunk_radius: float
    max_speed: float
    max_turn: float
    mount_angles: np.ndarray      # (2,)
    seg_lengths: np.ndarray       # (2,)
    limb_radius: float
    joint_lo: np.ndarray          # (2,) per joint [proximal, distal]
    joint_hi: np.ndarray
    max_joint_speed: float
    touch_range: float
    vision_rays: int
    vision_fov: float
    vision_range: float
    energy_initial: float
    basal: float
    effort_cost: float
    food_value: float
    low_threshold: float
    min_strength: float
    fatigue_decay: float
    fatigue_gain: float
    fatigue_max_penalty: float

    @property
    def reach(self) -> float:
        return float(self.seg_lengths.sum())

    @classmethod
    def from_config(cls, b: Config) -> "BodyParams":
        limits = np.array(b.joint_limits, dtype=float)
        return cls(
            trunk_radius=float(b.trunk_radius), max_speed=float(b.max_speed), max_turn=float(b.max_turn),
            mount_angles=np.array(b.limb_mount_angles, dtype=float),
            seg_lengths=np.array(b.segment_lengths, dtype=float),
            limb_radius=float(b.limb_radius),
            joint_lo=limits[:, 0], joint_hi=limits[:, 1],
            max_joint_speed=float(b.max_joint_speed), touch_range=float(b.touch_range),
            vision_rays=int(b.vision_rays), vision_fov=float(b.vision_fov), vision_range=float(b.vision_range),
            energy_initial=float(b.energy.initial), basal=float(b.energy.basal),
            effort_cost=float(b.energy.effort_cost), food_value=float(b.energy.food_value),
            low_threshold=float(b.energy.low_threshold), min_strength=float(b.energy.min_strength),
            fatigue_decay=float(b.fatigue.decay), fatigue_gain=float(b.fatigue.gain),
            fatigue_max_penalty=float(b.fatigue.max_penalty),
        )


@dataclass
class BodyState:
    pos: np.ndarray        # (2,) trunk centre
    heading: float         # radians, wrapped to (-pi, pi]
    q: np.ndarray          # (2, 2) joint angles [limb, joint]
    dq: np.ndarray         # (2, 2) joint change actually performed last step
    energy: float
    fatigue: float

    def copy(self) -> "BodyState":
        return BodyState(self.pos.copy(), float(self.heading), self.q.copy(), self.dq.copy(),
                         float(self.energy), float(self.fatigue))


def wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def limb_points(p: BodyParams, pos: np.ndarray, heading: float, q: np.ndarray) -> np.ndarray:
    """Points (2 limbs, [base, elbow, tip], xy) -> shape (2, 3, 2)."""
    a0 = heading + p.mount_angles                 # (2,)
    a1 = a0 + q[:, 0]
    a2 = a1 + q[:, 1]
    base = pos + p.trunk_radius * unit(a0)
    elbow = base + p.seg_lengths[0] * unit(a1)
    tip = elbow + p.seg_lengths[1] * unit(a2)
    return np.stack([base, elbow, tip], axis=1)


def _limb_inside(p: BodyParams, size: float, pos: np.ndarray, heading: float, q_limb: np.ndarray,
                 i: int) -> bool:
    a1 = heading + p.mount_angles[i] + q_limb[0]
    a2 = a1 + q_limb[1]
    base = pos + p.trunk_radius * unit(heading + p.mount_angles[i])
    elbow = base + p.seg_lengths[0] * unit(a1)
    tip = elbow + p.seg_lengths[1] * unit(a2)
    lo, hi = p.limb_radius, size - p.limb_radius
    pts = np.stack([elbow, tip])
    return bool(((pts >= lo) & (pts <= hi)).all())


def strength(p: BodyParams, energy: float, fatigue: float) -> float:
    s_energy = p.min_strength + (1.0 - p.min_strength) * min(1.0, energy / p.low_threshold)
    return s_energy * (1.0 - p.fatigue_max_penalty * fatigue)


def _trunk_clear(p: BodyParams, arena: Arena, pos: np.ndarray, tol: float = 1e-9) -> bool:
    """Trunk inside the arena and not overlapping any obstacle."""
    R = p.trunk_radius
    if (pos < R - tol).any() or (pos > arena.size - R + tol).any():
        return False
    if not len(arena.obstacle_r):
        return True
    gap = np.sqrt(((arena.obstacle_pos - pos) ** 2).sum(1)) - arena.obstacle_r - R
    return bool(gap.min() >= -tol)


def _resolve_trunk(p: BodyParams, arena: Arena, pos: np.ndarray, push_fraction: float) -> np.ndarray:
    R, size = p.trunk_radius, arena.size
    pos = push_out_of_circles(clamp_in_arena(pos, R, size), R, arena.obstacle_pos, arena.obstacle_r)
    if len(arena.neutral_r):
        diff = arena.neutral_pos - pos
        dist = np.sqrt((diff * diff).sum(1))
        overlap = R + arena.neutral_r - dist
        for j in np.flatnonzero(overlap > 0):
            n = diff[j] / dist[j] if dist[j] > 1e-9 else np.array([1.0, 0.0])
            r = arena.neutral_r[j]
            new = arena.neutral_pos[j] + n * overlap[j] * push_fraction
            new = push_out_of_circles(clamp_in_arena(new, r, size), r, arena.obstacle_pos, arena.obstacle_r)
            arena.neutral_pos[j] = clamp_in_arena(new, r, size)
    for _ in range(2):
        pos = push_out_of_circles(pos, R, arena.neutral_pos, arena.neutral_r)
        pos = push_out_of_circles(clamp_in_arena(pos, R, size), R, arena.obstacle_pos, arena.obstacle_r)
        pos = clamp_in_arena(pos, R, size)
    return pos


def move_body(p: BodyParams, arena: Arena, st: BodyState, applied: np.ndarray,
              push_fraction: float) -> tuple[np.ndarray, float]:
    """Advance the body one step in place. Returns (executed action, effort)."""
    executed = np.clip(applied, -1.0, 1.0) * strength(p, st.energy, st.fatigue)
    old_pos, old_h, old_q = st.pos.copy(), st.heading, st.q.copy()
    old_neutral = arena.neutral_pos.copy()

    h = old_h + executed[1] * p.max_turn
    pos = _resolve_trunk(p, arena, old_pos + executed[0] * p.max_speed * unit(h), push_fraction)

    q_new = np.clip(old_q + executed[2:].reshape(2, 2) * p.max_joint_speed, p.joint_lo, p.joint_hi)
    q = old_q.copy()
    # Wedged between obstacles, push-out may not fully converge: then the trunk stays put.
    trunk_ok = _trunk_clear(p, arena, pos)
    for i in range(2):
        if not trunk_ok:
            break
        if _limb_inside(p, arena.size, pos, h, q_new[i], i):
            q[i] = q_new[i]
        elif not _limb_inside(p, arena.size, pos, h, old_q[i], i):
            trunk_ok = False
    if not trunk_ok:
        pos, h = old_pos, old_h
        arena.neutral_pos[:] = old_neutral
        for i in range(2):
            q[i] = q_new[i] if _limb_inside(p, arena.size, pos, h, q_new[i], i) else old_q[i]

    st.pos, st.heading = pos, wrap_angle(h)
    st.dq = q - old_q
    st.q = q
    return executed, float(np.mean(executed**2))


def update_energy(p: BodyParams, st: BodyState, effort: float, n_eaten: int) -> dict[str, float]:
    """Apply one step of metabolism, effort cost and food gain; return the balance."""
    before = st.energy
    basal = p.basal
    effort_cost = p.effort_cost * effort
    food_gain = p.food_value * n_eaten
    unclipped = before - basal - effort_cost + food_gain
    st.energy = float(min(1.0, max(0.0, unclipped)))
    st.fatigue = float(min(1.0, max(0.0, p.fatigue_decay * st.fatigue + p.fatigue_gain * effort)))
    return {"before": before, "basal": basal, "effort_cost": effort_cost, "food_gain": food_gain,
            "unclipped": unclipped, "after": st.energy}
