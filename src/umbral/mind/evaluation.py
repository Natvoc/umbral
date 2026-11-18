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
    return {k: (float(s.sum()), int(c.sum())) for k, (s, c) in touch_event_sq_error_rows(layout, cur, nxt, pred).items()}


def touch_event_sq_error_rows(layout: ObsLayout, cur: torch.Tensor, nxt: torch.Tensor, pred: torch.Tensor
                              ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per event type: (per-row sums of per-sensor-step errors, per-row counts); rows = leading axis."""
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
    b = cur.shape[0]
    return {k: ((err * m).reshape(b, -1).sum(1).numpy(), m.reshape(b, -1).sum(1).numpy())
            for k, m in masks.items()}


def channel_sq_error_rows(layout: ObsLayout, pred: torch.Tensor, target: torch.Tensor
                          ) -> dict[str, tuple[np.ndarray, int]]:
    """Per channel: (per-row sums of squared errors, elements per row); rows = leading axis."""
    b = pred.shape[0]
    out = {}
    for ch in CHANNELS:
        d = pred[..., layout.slices[ch]].double() - target[..., layout.slices[ch]].double()
        out[ch] = ((d * d).reshape(b, -1).sum(1).numpy(), d[0].numel())
    return out


def evaluate(models: Mapping[str, WorldModel], config: Config, histories: list[str], seeds: list[int],
             policy_seeds: list[int], steps: int, chunk: int) -> dict:
    """Run the OU policy on fixed seeds; accumulate errors of each model and of persistence.

    Returns ``{"mse": {pred: {history: {channel: [sum, n]}}}, "touch": {pred: {event: [sum, n]}},
    "rows": {...}}``. ``rows`` keeps the same sums per environment (one evaluation seed each):
    ``rows["mse"][pred][channel]`` and ``rows["touch"][model][event]`` are ``(n_envs, 2)`` arrays
    of [sum, count], used for the bootstrap over seeds; ``rows["histories"]`` labels the rows.
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
    mse_rows = {p: {c: np.zeros((n, 2)) for c in CHANNELS} for p in predictors}
    touch_rows = {p: {e: np.zeros((n, 2)) for e in TOUCH_EVENTS} for p in models}
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
                for e, (s, cnt) in touch_event_sq_error_rows(env.layout, cur, nxt, preds[k]).items():
                    touch_rows[k][e][:, 0] += s
                    touch_rows[k][e][:, 1] += cnt
        for p, pred in preds.items():
            for c, (s, cnt) in channel_sq_error_rows(env.layout, pred, nxt).items():
                mse_rows[p][c][:, 0] += s
                mse_rows[p][c][:, 1] += cnt
        done += T
    hids = sorted(set(histories))
    rows_of = {hid: np.array([i for i, x in enumerate(histories) if x == hid]) for hid in hids}
    mse = {p: {hid: {c: [float(mse_rows[p][c][r, 0].sum()), int(mse_rows[p][c][r, 1].sum())]
                     for c in CHANNELS} for hid, r in rows_of.items()} for p in predictors}
    touch = {p: {e: [float(v[:, 0].sum()), int(v[:, 1].sum())] for e, v in ev.items()}
             for p, ev in touch_rows.items()}
    return {"mse": mse, "touch": touch,
            "rows": {"mse": mse_rows, "touch": touch_rows, "histories": list(histories)}}


def bootstrap_ratio_ci(num: np.ndarray, den: np.ndarray, histories: list[str], n_boot: int, seed: int,
                       level: float = 0.95) -> tuple[float, float]:
    """Percentile CI of ``mean(num) / mean(den)`` resampling evaluation seeds within each history.

    ``num`` and ``den`` are ``(n_envs, 2)`` [sum, count] arrays over the same rows (seeds). Each
    resample draws, inside every history, as many rows as it has, with replacement (stratified
    bootstrap), and recomputes both pooled means from the drawn rows.
    """
    rng = np.random.default_rng(seed)
    hist = np.asarray(histories)
    idx = np.concatenate([rng.choice(np.flatnonzero(hist == h), size=(n_boot, int((hist == h).sum())))
                          for h in sorted(set(histories))], axis=1)
    ns, nc = num[idx, 0].sum(1), num[idx, 1].sum(1)
    ds, dc = den[idx, 0].sum(1), den[idx, 1].sum(1)
    ok = (nc > 0) & (dc > 0) & (ds > 0)
    if not ok.any():
        return float("nan"), float("nan")
    r = (ns[ok] / nc[ok]) / (ds[ok] / dc[ok])
    a = (1 - level) / 2
    return float(np.quantile(r, a)), float(np.quantile(r, 1 - a))


def ratio_cis(sums: dict, n_boot: int = 10_000, seed: int = 0) -> dict:
    """95 % bootstrap CIs for every ratio of the evaluation tables.

    ``mse``: efference/persistence and efference/no_efference per channel; ``touch_events``:
    efference/no_efference per event.
    """
    rows, hist = sums["rows"], sums["rows"]["histories"]
    out = {"n_boot": n_boot, "seed": seed, "level": 0.95, "mse": {}, "touch_events": {}}
    for c in CHANNELS:
        m = rows["mse"]
        out["mse"][c] = {
            "efference/persistence": bootstrap_ratio_ci(m["efference"][c], m["persistence"][c], hist, n_boot, seed),
            "efference/no_efference": bootstrap_ratio_ci(m["efference"][c], m["no_efference"][c], hist, n_boot,
                                                         seed),
        }
    for e in TOUCH_EVENTS:
        t = rows["touch"]
        out["touch_events"][e] = {"efference/no_efference": bootstrap_ratio_ci(
            t["efference"][e], t["no_efference"][e], hist, n_boot, seed)}
    return out


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
