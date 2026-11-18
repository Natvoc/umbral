"""Internal drives (Phase 2b): the only source of reward. The environment's reward is always 0.

All three are computed inside the mind, from what reaches it through the BodyLink
(the observation) and from its own intended action. None reads the body or the
world directly.

- **Hunger** (the same for every temperament): homeostatic drive reduction. With
  ``D(E) = (E* - E)**2`` and the observed energy ``E`` (interoception), the reward of
  step t is ``hunger_scale * (D(E_t) - D(E_{t+1}))``: positive when energy moves
  towards the set point ``E*``, negative when it moves away, and no larger than
  that change.
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

The reward is ``hunger + w_curiosity * curiosity_scale * curiosity + w_caution * caution``.
The weights ``w_*`` define the temperament. A common running scale (the std of the
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


def hunger_reward(e_now: np.ndarray, e_next: np.ndarray, set_point: float, scale: float) -> np.ndarray:
    """Drive reduction: ``scale * (D(E_t) - D(E_{t+1}))``."""
    return scale * (homeostatic_drive(e_now, set_point) - homeostatic_drive(e_next, set_point))


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
        self.ema_rate = float(drives_cfg.curiosity_ema)
        self.effort_cost = float(effort_cost)
        self.error_mean: float | None = None  # running mean of the raw curiosity error

    def __call__(self, cur: np.ndarray, nxt: np.ndarray, pred: np.ndarray, action: np.ndarray
                 ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Rewards (B, T) for transitions ``o_t -> o_{t+1}`` with intended actions ``a_t``.

        ``pred`` is the mind's prediction of ``o_{t+1}``. The running mean is updated
        once per call, with the batch mean, *after* normalising this batch.
        """
        e = self.layout.slices["intero"].start  # energy is the first interoceptive value
        hunger = hunger_reward(cur[..., e], nxt[..., e], self.set_point, self.hunger_scale)
        err = world_prediction_error(self.layout, cur, nxt, pred)
        batch_mean = float(err.mean())
        if self.error_mean is None:
            self.error_mean = batch_mean
        curiosity = err / max(self.error_mean, 1e-12)
        self.error_mean = (1 - self.ema_rate) * self.error_mean + self.ema_rate * batch_mean
        caution = caution_reward(action, self.effort_cost, self.caution_scale)
        t = self.temperament
        weighted = {"hunger": hunger, "curiosity": t.curiosity * self.curiosity_scale * curiosity,
                    "caution": t.caution * caution}
        total = weighted["hunger"] + weighted["curiosity"] + weighted["caution"]
        return total, {"hunger": hunger, "curiosity": curiosity, "caution": caution, "curiosity_raw": err,
                       "weighted": weighted}

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
