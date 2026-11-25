"""Internal drives (Phase 2b): the only source of reward. The environment's reward is always 0.

All three are computed inside the mind, from what reaches it through the BodyLink
(the observation) and from its own intended action. None reads the body or the
world directly.

- **Hunger** (the same for every temperament): homeostatic drive reduction. With
  ``D(E) = (E* - E)**2`` and the observed energy ``E`` (interoception), the reward of
  step t is ``hunger_scale * (D(E_t) - D(E_{t+1}))``: positive when energy moves
  towards the set point ``E*``, negative when it moves away, and no larger than
  that change.

  **Energy debt** (``drives.hunger_debt``, since 25 Nov 2026; before it, hunger v1).
  The physical energy has its floor at 0, so with ``E`` alone the drive stopped at 0:
  ``D`` could not rise any more, the punishment of every step ended there, and reaching
  0 worked as a relief (the 22 Nov diagnosis: the critic valued energy 0 *above* energy
  just over it). Hunger now uses an internal energy ``E_int``, which can be negative:
  ``E_int = E`` while ``E > 0`` (exactly as before); at ``E = 0`` it keeps falling every
  step by the cost the body would have paid (effective basal metabolism plus the cost of
  the effort executed), down to ``drives.debt_floor``; eating sets it back to the
  physical energy after eating (the food settles the debt). Hunger is then
  ``hunger_scale * (D(E_int,t) - D(E_int,t+1))``: at 0 the signal goes on as just before
  reaching 0, with no false relief. ``E_int`` is used for this reward only: the mind
  does not observe it (interoception is still the physical energy), and the curiosity
  gate still uses the observed energy. The cost per step comes from the body's energy
  balance: it is the experimenter's reward function, never part of the observation.
- **Curiosity**: the prediction error of the mind's own model on the next
  observation, divided by a running mean (exponential moving average) of that
  error, so its typical size is 1. Only channels that look at the *world* count:
  vision, and the touch sensors whose contact (now or next) is not with the own
  body. Proprioception and self-contact are excluded on purpose. Otherwise the
  agent could make its error "interesting" by flailing its limbs without control,
  which is its own body, not novelty in the world.
- **Caution**: a discomfort proportional to the energy the step costs through
  effort, ``-caution_scale * effort_cost * mean(a_t**2)``. It uses the *intended*
  action ``a_t``, the one the mind knows. The executed action, which may differ by
  joint limits and strength, belongs to the body.

**Hunger gates curiosity** (shared architecture, the same for every temperament).
With the observed energy ``E_t``::

    g(E) = clip((E - E_low) / (E* - E_low), 0, 1)        (E_low = drives.gate_low)

curiosity is multiplied by ``g(E_t)``; hunger and caution are not. With high energy
each temperament is expressed in full; as energy falls, curiosity fades for everyone
and hunger rules, as an urgent need overrides exploration in animals. Reason (18 Nov
2026 calibration): curiosity, normalised by its running mean, was always "on" at the
same strength, and hunger was felt only when it was already late: the curious agent
exhausted itself.

Caution is *not* gated. The first version (19 Nov 2026) gated curiosity and caution
alike, and failed: caution is the brake that protects energy (the life with caution
at 0 had shown it), so gating it released the brake exactly when energy fell. The
curious agent's effort rose from 0.20 to 0.28 as ``g`` fell to 0.5, and both seeds
broke the ethics rule at a caution scale where, ungated, it had stayed healthy.

The reward is ``hunger + g(E) * w_curiosity * curiosity_scale * curiosity + w_caution * caution``.
The weights ``w_*`` define the temperament (the proportion of curiosity and caution). A common running scale (the std of the
discounted return, ``RewardScaler``) divides the total. It is one factor for all
drives, so it does not change their proportions.

Note for Phase 5 (the loss of the body), not acted on now: without a body, caution
keeps acting on the void actions (they are intended actions), and curiosity keeps
acting on a prediction error that no longer has a real signal behind it. How the
drives behave during the loss, and whether learning continues or freezes, is to be
decided in the Phase 5 protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from umbral.body.sensors import N_TOUCH, TOUCH_WIDTH, ObsLayout
from umbral.world.arena import SELF

SELF_COL = 1 + SELF


@dataclass(frozen=True)
class Temperament:
    name: str
    curiosity: float
    caution: float


def temperament_from_config(config, name: str) -> Temperament:
    t = config.temperaments[name]
    return Temperament(name, float(t.curiosity), float(t.caution))


def homeostatic_drive(energy: np.ndarray, set_point: float) -> np.ndarray:
    return (set_point - energy) ** 2


def hunger_gate(energy: np.ndarray, set_point: float, low: float) -> np.ndarray:
    """``g(E) = clip((E - low) / (set_point - low), 0, 1)``: 1 when sated, 0 when hungry."""
    return np.clip((energy - low) / (set_point - low), 0.0, 1.0)


def hunger_reward(e_now: np.ndarray, e_next: np.ndarray, set_point: float, scale: float) -> np.ndarray:
    """Drive reduction: ``scale * (D(E_t) - D(E_{t+1}))``."""
    return scale * (homeostatic_drive(e_now, set_point) - homeostatic_drive(e_next, set_point))


def internal_energy_step(e_int: np.ndarray, e_next: np.ndarray, cost: np.ndarray, ate: np.ndarray,
                         floor: float) -> np.ndarray:
    """``E_int`` after one step (all (N,)): the physical energy while it is above 0 or after eating,
    otherwise the previous ``E_int`` minus the step's cost, never below ``floor``."""
    debt = np.maximum(float(floor), e_int - cost)
    return np.where((e_next > 0.0) | ate, e_next, debt)


def caution_reward(action: np.ndarray, effort_cost: float, scale: float) -> np.ndarray:
    """``-scale * effort_cost * mean(a**2)`` over the last axis (intended action)."""
    return -scale * effort_cost * np.mean(action * action, axis=-1)


def world_prediction_error(layout: ObsLayout, cur: np.ndarray, nxt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Mean squared error over the world-facing parts: vision, and touch sensors without self-contact.

    A touch sensor counts only if neither its current nor its next contact type is
    ``self``. Arrays are (..., 107); returns (...).
    """
    lead = cur.shape[:-1]
    vs, ts = layout.slices["vision"], layout.slices["touch"]
    v_err = ((pred[..., vs] - nxt[..., vs]) ** 2).sum(-1)
    c = cur[..., ts].reshape(*lead, N_TOUCH, TOUCH_WIDTH)
    n = nxt[..., ts].reshape(*lead, N_TOUCH, TOUCH_WIDTH)
    p = pred[..., ts].reshape(*lead, N_TOUCH, TOUCH_WIDTH)
    world = (c[..., SELF_COL] != 1) & (n[..., SELF_COL] != 1)
    t_err = (((p - n) ** 2).sum(-1) * world).sum(-1)
    count = (vs.stop - vs.start) + TOUCH_WIDTH * world.sum(-1)
    return (v_err + t_err) / count


class Drives:
    """Computes the drive rewards of a rollout; keeps the curiosity running mean (checkpointed)."""

    def __init__(self, layout: ObsLayout, drives_cfg, temperament: Temperament, effort_cost: float):
        self.layout = layout
        self.temperament = temperament
        self.set_point = float(drives_cfg.set_point)
        self.hunger_scale = float(drives_cfg.hunger_scale)
        self.curiosity_scale = float(drives_cfg.curiosity_scale)
        self.caution_scale = float(drives_cfg.caution_scale)
        self.gate_low = float(drives_cfg.gate_low)
        self.ema_rate = float(drives_cfg.curiosity_ema)
        self.effort_cost = float(effort_cost)
        self.hunger_debt = bool(drives_cfg.get("hunger_debt", False))  # configs before 25 Nov 2026: hunger v1
        self.debt_floor = float(drives_cfg.get("debt_floor", -1.0))
        self.error_mean: float | None = None  # running mean of the raw curiosity error

    def __call__(self, cur: np.ndarray, nxt: np.ndarray, pred: np.ndarray, action: np.ndarray,
                 e_int: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Rewards (B, T) for transitions ``o_t -> o_{t+1}`` with intended actions ``a_t``.

        ``pred`` is the mind's prediction of ``o_{t+1}``. The running mean is updated
        once per call, with the batch mean, *after* normalising this batch. With
        ``hunger_debt``, ``e_int`` (B, T+1) is the internal energy at every observation
        and hunger uses it; without it (or with ``e_int`` None) hunger uses the observed energy.
        """
        e = self.layout.slices["intero"].start  # energy is the first interoceptive value
        if self.hunger_debt and e_int is not None:
            hunger = hunger_reward(e_int[..., :-1], e_int[..., 1:], self.set_point, self.hunger_scale)
        else:
            hunger = hunger_reward(cur[..., e], nxt[..., e], self.set_point, self.hunger_scale)
        err = world_prediction_error(self.layout, cur, nxt, pred)
        batch_mean = float(err.mean())
        if self.error_mean is None:
            self.error_mean = batch_mean
        curiosity = err / max(self.error_mean, 1e-12)
        self.error_mean = (1 - self.ema_rate) * self.error_mean + self.ema_rate * batch_mean
        caution = caution_reward(action, self.effort_cost, self.caution_scale)
        t = self.temperament
        gate = hunger_gate(cur[..., e], self.set_point, self.gate_low)  # energy the mind observes now
        weighted = {"hunger": hunger, "curiosity": gate * t.curiosity * self.curiosity_scale * curiosity,
                    "caution": t.caution * caution}  # caution is never gated: it is the brake
        total = weighted["hunger"] + weighted["curiosity"] + weighted["caution"]
        return total, {"hunger": hunger, "curiosity": curiosity, "caution": caution, "curiosity_raw": err,
                       "gate": gate, "weighted": weighted}

    def get_state(self) -> dict[str, Any]:
        return {"error_mean": self.error_mean}

    def set_state(self, state: dict[str, Any]) -> None:
        self.error_mean = None if state["error_mean"] is None else float(state["error_mean"])


class RewardScaler:
    """Divides rewards by the running std of the discounted return (one common factor)."""

    def __init__(self, n_envs: int, gamma: float):
        self.gamma = float(gamma)
        self.ret = np.zeros(n_envs)
        self.count, self.mean, self.m2 = 0, 0.0, 0.0

    def __call__(self, rewards: np.ndarray) -> np.ndarray:
        """rewards (N, T) in time order; updates the statistics and returns the scaled rewards."""
        for t in range(rewards.shape[1]):
            self.ret = self.gamma * self.ret + rewards[:, t]
            nb, mb = len(self.ret), float(self.ret.mean())
            m2b = float(((self.ret - mb) ** 2).sum())
            tot = self.count + nb
            d = mb - self.mean
            self.mean += d * nb / tot
            self.m2 += m2b + d * d * self.count * nb / tot
            self.count = tot
        return rewards / self.std

    @property
    def std(self) -> float:
        if self.count < 2:
            return 1.0
        return max(float(np.sqrt(self.m2 / (self.count - 1))), 1e-8)

    def reset_rows(self, rows: np.ndarray) -> None:
        self.ret[rows] = 0.0

    def get_state(self) -> dict[str, Any]:
        return {"ret": self.ret.copy(), "count": self.count, "mean": self.mean, "m2": self.m2}

    def set_state(self, state: dict[str, Any]) -> None:
        self.ret = np.array(state["ret"], dtype=float)
        self.count, self.mean, self.m2 = int(state["count"]), float(state["mean"]), float(state["m2"])
