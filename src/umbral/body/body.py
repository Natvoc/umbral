"""The physical body: trunk, two 2-segment limbs, kinematics, energy and fatigue.

Pure kinematics in numpy, no physics engine. The kernels work on batches
(leading axis ``B``); the unbatched helpers (``limb_points``, ``move_body``,
``update_energy``) are thin wrappers around a batch of one, so there is a single
implementation.

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
  overlap, e.g. wedged between two obstacles, the trunk move is reverted).
  Neutral objects absorb ``push_fraction`` of an overlap (a light push) and never
  pass through walls or obstacles themselves.
- Limbs only sense obstacles, but may not cross walls: a joint move that would put
  a limb point outside the arena is reverted; if the limb is outside even with its
  previous angles (trunk moved too close to a wall), the trunk move is reverted.
- A reverted trunk move reverts only the translation: the body still turns in place,
  unless the new heading would put a limb outside the arena with both its new and its
  previous angles (then the turn is reverted too). Since 22 Nov 2026: before, the turn
  was always reverted with the translation, and bodies stayed wedged for over 1,000
  steps repeating the same blocked action (docs/protocolo.md).
- Energy: ``E' = clip(E - basal - effort_cost * effort + food_value * n_eaten, 0, 1)``
  with ``effort = mean(executed ** 2)``. Energy 0 means weaker, slower movement,
  never death.
- Fatigue (provisional): ``F' = clip(decay * F + gain * effort, 0, 1)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from umbral.config import Config
from umbral.world.arena import Arena, clamp_in_arena, norm2, push_out_of_circles, unit, wrap_angle

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
    """One body (unbatched)."""
    pos: np.ndarray        # (2,) trunk centre
    heading: float         # radians, wrapped to [-pi, pi)
    q: np.ndarray          # (2, 2) joint angles [limb, joint]
    dq: np.ndarray         # (2, 2) joint change actually performed last step
    energy: float
    fatigue: float

    def copy(self) -> "BodyState":
        return BodyState(self.pos.copy(), float(self.heading), self.q.copy(), self.dq.copy(),
                         float(self.energy), float(self.fatigue))


# --- batched kernels ------------------------------------------------------------

def limb_points_b(p: BodyParams, pos: np.ndarray, heading: np.ndarray, q: np.ndarray) -> np.ndarray:
    """pos (B,2), heading (B,), q (B,2,2) -> points (B, 2 limbs, [base, elbow, tip], 2)."""
    a0 = heading[:, None] + p.mount_angles[None, :]      # (B, 2)
    a1 = a0 + q[:, :, 0]
    a2 = a1 + q[:, :, 1]
    base = pos[:, None, :] + p.trunk_radius * unit(a0)
    elbow = base + p.seg_lengths[0] * unit(a1)
    tip = elbow + p.seg_lengths[1] * unit(a2)
    return np.stack([base, elbow, tip], axis=2)


def _limbs_inside_b(p: BodyParams, size: float, pos: np.ndarray, heading: np.ndarray,
                    q: np.ndarray) -> np.ndarray:
    """(B, 2) bool: elbow and tip of each limb inside the arena (minus limb radius)."""
    pts = limb_points_b(p, pos, heading, q)[:, :, 1:]    # (B, 2, 2 points, 2)
    lo, hi = p.limb_radius, size - p.limb_radius
    return ((pts >= lo) & (pts <= hi)).all(axis=(-1, -2))


def strength_b(p: BodyParams, energy: np.ndarray, fatigue: np.ndarray) -> np.ndarray:
    s_energy = p.min_strength + (1.0 - p.min_strength) * np.minimum(1.0, energy / p.low_threshold)
    return s_energy * (1.0 - p.fatigue_max_penalty * fatigue)


def _trunk_clear_b(p: BodyParams, size: float, pos: np.ndarray, obs_pos: np.ndarray, obs_r: np.ndarray,
                   tol: float = 1e-9) -> np.ndarray:
    R = p.trunk_radius
    ok = ((pos >= R - tol) & (pos <= size - R + tol)).all(axis=-1)
    if obs_pos.shape[1]:
        gap = norm2(obs_pos - pos[:, None, :]) - obs_r - R
        ok &= gap.min(axis=-1) >= -tol
    return ok


def _resolve_trunk_b(p: BodyParams, size: float, pos: np.ndarray, obs_pos: np.ndarray, obs_r: np.ndarray,
                     neu_pos: np.ndarray, neu_r: np.ndarray, push_fraction: float
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Returns (new trunk pos (B,2), new neutral positions (B,N,2))."""
    R = p.trunk_radius
    pos = push_out_of_circles(clamp_in_arena(pos, R, size)[:, None], R, obs_pos, obs_r)[:, 0]
    neu = neu_pos
    if neu_pos.shape[1]:
        diff = neu_pos - pos[:, None, :]
        dist = norm2(diff)
        overlap = R + neu_r - dist
        hit = overlap > 0
        if hit.any():  # rows without a touched neutral keep neu_pos unchanged either way
            tiny = dist <= 1e-9
            safe = np.where(tiny, 1.0, dist)
            n = np.where(tiny[..., None], np.array([1.0, 0.0]), diff / safe[..., None])
            moved = neu_pos + n * (overlap * push_fraction)[..., None]
            moved = push_out_of_circles(clamp_in_arena(moved, neu_r, size), neu_r, obs_pos, obs_r)
            moved = clamp_in_arena(moved, neu_r, size)
            neu = np.where(hit[..., None], moved, neu_pos)
    for _ in range(2):
        pos = push_out_of_circles(pos[:, None], R, neu, neu_r)[:, 0]
        pos = push_out_of_circles(clamp_in_arena(pos, R, size)[:, None], R, obs_pos, obs_r)[:, 0]
        pos = clamp_in_arena(pos, R, size)
    return pos, neu


def move_body_b(p: BodyParams, size: float, pos: np.ndarray, heading: np.ndarray, q: np.ndarray,
                energy: np.ndarray, fatigue: np.ndarray, obs_pos: np.ndarray, obs_r: np.ndarray,
                neu_pos: np.ndarray, neu_r: np.ndarray, applied: np.ndarray, push_fraction: float) -> dict:
    """One movement step for a batch. Pure: returns new arrays, mutates nothing.

    Keys: pos, heading, q, dq, neutral_pos, executed, effort.
    """
    executed = np.clip(applied, -1.0, 1.0) * strength_b(p, energy, fatigue)[:, None]
    h_new = heading + executed[:, 1] * p.max_turn
    pos_try = pos + (executed[:, 0] * p.max_speed)[:, None] * unit(h_new)
    pos_new, neu_new = _resolve_trunk_b(p, size, pos_try, obs_pos, obs_r, neu_pos, neu_r, push_fraction)

    q_new = np.clip(q + executed[:, 2:].reshape(-1, 2, 2) * p.max_joint_speed, p.joint_lo, p.joint_hi)
    ok_new = _limbs_inside_b(p, size, pos_new, h_new, q_new)
    ok_old = _limbs_inside_b(p, size, pos_new, h_new, q)
    trunk_ok = _trunk_clear_b(p, size, pos_new, obs_pos, obs_r) & (ok_new | ok_old).all(axis=-1)
    # translation reverted: turn in place if the new heading keeps every limb inside
    ok_turn_new = _limbs_inside_b(p, size, pos, h_new, q_new)
    ok_turn_old = _limbs_inside_b(p, size, pos, h_new, q)
    turn_ok = ~trunk_ok & (ok_turn_new | ok_turn_old).all(axis=-1)
    ok_reverted = _limbs_inside_b(p, size, pos, heading, q_new)

    q_keep = np.where(ok_new[..., None], q_new, q)
    q_turn = np.where(ok_turn_new[..., None], q_new, q)
    q_rev = np.where(ok_reverted[..., None], q_new, q)
    q_out = np.where(trunk_ok[:, None, None], q_keep, np.where(turn_ok[:, None, None], q_turn, q_rev))
    return {
        "pos": np.where(trunk_ok[:, None], pos_new, pos),
        "heading": wrap_angle(np.where(trunk_ok | turn_ok, h_new, heading)),
        "q": q_out,
        "dq": q_out - q,
        "neutral_pos": np.where(trunk_ok[:, None, None], neu_new, neu_pos),
        "executed": executed,
        "effort": np.mean(executed * executed, axis=-1),
    }


def update_energy_b(p: BodyParams, energy: np.ndarray, fatigue: np.ndarray, effort: np.ndarray,
                    n_eaten: np.ndarray, metabolism: float = 1.0) -> dict[str, np.ndarray]:
    """Energy/fatigue after one step, with the balance terms (all (B,)).

    ``metabolism`` scales the whole energy cost of living (basal and effort); it is 1
    except during the protected childhood of Phase 2b (``life.protected_infancy``).
    """
    basal = np.full(energy.shape, p.basal * metabolism)
    effort_cost = p.effort_cost * metabolism * effort
    food_gain = p.food_value * n_eaten
    unclipped = energy - basal - effort_cost + food_gain
    return {
        "before": np.array(energy, dtype=float, copy=True), "basal": basal, "effort_cost": effort_cost, "food_gain": food_gain,
        "unclipped": unclipped, "after": np.minimum(1.0, np.maximum(0.0, unclipped)),
        "fatigue": np.minimum(1.0, np.maximum(0.0, p.fatigue_decay * fatigue + p.fatigue_gain * effort)),
    }


# --- unbatched wrappers (batch of one) -------------------------------------------

def limb_points(p: BodyParams, pos: np.ndarray, heading: float, q: np.ndarray) -> np.ndarray:
    """Points (2 limbs, [base, elbow, tip], xy) -> shape (2, 3, 2)."""
    return limb_points_b(p, np.asarray(pos, float)[None], np.array([heading], float),
                         np.asarray(q, float)[None])[0]


def strength(p: BodyParams, energy: float, fatigue: float) -> float:
    return float(strength_b(p, np.array([energy]), np.array([fatigue]))[0])


def move_body(p: BodyParams, arena: Arena, st: BodyState, applied: np.ndarray,
              push_fraction: float) -> tuple[np.ndarray, float]:
    """Advance one body in place (and push the arena's neutral objects). Returns (executed, effort)."""
    out = move_body_b(
        p, arena.size, st.pos[None], np.array([st.heading]), st.q[None],
        np.array([st.energy]), np.array([st.fatigue]),
        arena.obstacle_pos[None], arena.obstacle_r[None], arena.neutral_pos[None], arena.neutral_r[None],
        np.asarray(applied, float)[None], push_fraction)
    st.pos, st.heading = out["pos"][0], float(out["heading"][0])
    st.q, st.dq = out["q"][0], out["dq"][0]
    arena.neutral_pos[:] = out["neutral_pos"][0]
    return out["executed"][0], float(out["effort"][0])


def update_energy(p: BodyParams, st: BodyState, effort: float, n_eaten: int) -> dict[str, float]:
    """Apply one step of metabolism, effort cost and food gain in place; return the balance."""
    b = update_energy_b(p, np.array([st.energy]), np.array([st.fatigue]), np.array([effort]),
                        np.array([n_eaten], float))
    st.energy, st.fatigue = float(b["after"][0]), float(b["fatigue"][0])
    return {k: float(b[k][0]) for k in ("before", "basal", "effort_cost", "food_gain", "unclipped", "after")}
