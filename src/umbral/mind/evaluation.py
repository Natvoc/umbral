"""Evaluation of predictive minds on fixed, unseen seeds.

Metrics (all in observation space):

- ``mse``: per-channel mean squared error over every step (the general metric;
  includes persistence as a baseline).
- ``touch events`` (pre-specified before the long Phase 2a run): error only on the
  sensor-steps where contact *starts* (onset: intensity 0 at t, > 0 at t+1) or
  *ends* (offset: > 0 at t, 0 at t+1). The error of a sensor-step is the mean
  squared error over its 6 values (intensity + type). Each event type is also
  split into **self**-contact (the contact involved is with the own body: the
  type at t+1 for onsets, at t for offsets) and **external** contact. These are
  compared between the efference and no-efference models only: persistence is
  wrong on every change by definition.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch

from umbral.body.sensors import N_TOUCH, TOUCH_WIDTH, ObsLayout
from umbral.config import Config
from umbral.mind.model import CHANNELS, WorldModel, channel_sq_error
from umbral.mind.policy import OUPolicy
from umbral.world.arena import SELF
from umbral.world.batched import BatchedUmbralEnv

TOUCH_EVENTS = ("onset", "onset_self", "onset_external", "offset", "offset_self", "offset_external")
SELF_COL = 1 + SELF  # column of the "self" one-hot in a touch row


def touch_event_sq_error(layout: ObsLayout, cur: torch.Tensor, nxt: torch.Tensor, pred: torch.Tensor
                         ) -> dict[str, tuple[float, int]]:
    """Per event type: (sum of per-sensor-step errors, number of sensor-steps)."""
    sl = layout.slices["touch"]
    lead = cur.shape[:-1]
    c = cur[..., sl].reshape(*lead, N_TOUCH, TOUCH_WIDTH).double()
    n = nxt[..., sl].reshape(*lead, N_TOUCH, TOUCH_WIDTH).double()
    p = pred[..., sl].reshape(*lead, N_TOUCH, TOUCH_WIDTH).double()
    err = ((p - n) ** 2).mean(-1)
    onset = (c[..., 0] == 0) & (n[..., 0] > 0)
    offset = (c[..., 0] > 0) & (n[..., 0] == 0)
    self_next, self_cur = n[..., SELF_COL] == 1, c[..., SELF_COL] == 1
    masks = {
        "onset": onset, "onset_self": onset & self_next, "onset_external": onset & ~self_next,
        "offset": offset, "offset_self": offset & self_cur, "offset_external": offset & ~self_cur,
    }
    return {k: (float(err[m].sum()), int(m.sum())) for k, m in masks.items()}


def evaluate(models: Mapping[str, WorldModel], config: Config, histories: list[str], seeds: list[int],
             policy_seeds: list[int], steps: int, chunk: int) -> dict:
    """Run the OU policy on fixed seeds; accumulate errors of each model and of persistence.

    Returns ``{"mse": {pred: {history: {channel: [sum, n]}}}, "touch": {pred: {event: [sum, n]}}}``.
    Pure with respect to the models and the global RNGs (own env, own generators, no grad).
    """
    n = len(histories)
    env = BatchedUmbralEnv(config, n, histories)
    env.reset(seed=list(seeds))
    tc = config.train
    policy = OUPolicy(n, float(tc.smoothing), float(tc.scale), list(policy_seeds))
    h = {k: m.initial_state(n) for k, m in models.items()}
    prev_action = np.zeros((n, 6))
    predictors = (*models, "persistence")
    hids = sorted(set(histories))
    rows_of = {hid: np.array([i for i, x in enumerate(histories) if x == hid]) for hid in hids}
    mse = {p: {hid: {c: [0.0, 0] for c in CHANNELS} for hid in hids} for p in predictors}
    touch = {p: {e: [0.0, 0] for e in TOUCH_EVENTS} for p in models}
    done = 0
    while done < steps:
        T = min(chunk, steps - done)
        obs = np.empty((n, T + 1, env.observation_dim))
        act = np.empty((n, T, 6))
        prev = np.empty((n, T, 6))
        obs[:, 0] = env.observation()["flat"]
        for t in range(T):
            o, _, _, _, info = env.step(policy.act())
            prev[:, t], act[:, t] = prev_action, info["action_intended"]
            obs[:, t + 1] = o["flat"]
            prev_action = info["action_intended"]
        cur = torch.as_tensor(obs[:, :-1], dtype=torch.float32)
        nxt = torch.as_tensor(obs[:, 1:], dtype=torch.float32)
        pa, aa = torch.as_tensor(prev, dtype=torch.float32), torch.as_tensor(act, dtype=torch.float32)
        preds = {"persistence": cur}
        with torch.no_grad():
            for k, m in models.items():
                out = m(cur, pa, aa, h[k])
                h[k] = out.h_last
                preds[k] = m.predict_next(cur, out)
                for e, (s, cnt) in touch_event_sq_error(env.layout, cur, nxt, preds[k]).items():
                    touch[k][e][0] += s
                    touch[k][e][1] += cnt
        for p, pred in preds.items():
            for hid, rows in rows_of.items():
                for c, (s, cnt) in channel_sq_error(env.layout, pred[rows], nxt[rows]).items():
                    mse[p][hid][c][0] += s
                    mse[p][hid][c][1] += cnt
        done += T
    return {"mse": mse, "touch": touch}


def summarize(sums: dict) -> dict:
    """Means from ``evaluate`` sums: mse per channel (all histories and per history), touch events."""
    mse = sums["mse"]
    preds = list(mse)
    hids = list(next(iter(mse.values())))

    def mean(p: str, keys: list[str], c: str) -> float:
        s = sum(mse[p][k][c][0] for k in keys)
        n = sum(mse[p][k][c][1] for k in keys)
        return s / n

    return {
        "mse": {c: {p: mean(p, hids, c) for p in preds} for c in CHANNELS},
        "mse_by_history": {h: {c: {p: mean(p, [h], c) for p in preds} for c in CHANNELS} for h in hids},
        "touch_events": {e: {p: (v[e][0] / v[e][1] if v[e][1] else None) for p, v in sums["touch"].items()}
                         for e in TOUCH_EVENTS},
        "touch_event_counts": {e: next(iter(sums["touch"].values()))[e][1] for e in TOUCH_EVENTS},
    }
