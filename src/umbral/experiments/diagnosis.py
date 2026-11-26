"""``zero_diagnosis``: what happens to an agent's drives and bodies at energy 0 (informative).

From a life checkpoint (``diagnosis.checkpoint``: world, mind state, policy noise and
drives exactly as they were), the agent's policy runs ``diagnosis.steps`` more steps on
its ``life.n_envs`` bodies **without learning**, in the checkpoint's own configuration.
It reports (``diagnosis.json``):

- by energy band (exactly 0, (0, 0.05), > 0.2): each drive's weighted reward per step,
  the total, the critic's value, the action and the effort;
- how often the policy saw low energy during training (energy histogram of the
  checkpoint's ethics record, 0.01 bins);
- per body with a stretch at energy 0 of at least ``ethics.zero_run_limit`` steps: the
  distance to the nearest active food at the start of the stretch, the straight-line
  steps it would need at its strength there (turning plus walking until a limb tip
  reaches the food), and what it did instead (path, net approach, turning);
- per body touching food without eating: which touch sensor touches it, how far the
  limb tips are, and whether moving only the joints (trunk still) could reach it.

Hunger with and without the energy debt (25 Nov 2026): on the same steps, ``hunger_v1``
(the observed energy, the drive before the correction) and ``hunger_debt`` (the internal
energy ``E_int``, ``umbral.mind.drives``), whatever the checkpoint's own configuration
used (``hunger`` is the one it used). ``E_int`` starts from the checkpoint's record when
it has one; older checkpoints have none, so bodies already at 0 start without debt
(``debt_start``), which can only make the debt look smaller.

Checkpoints of the common feeding phase (``extra["history"] == "all"``, 26 Nov 2026) are
diagnosed with their own mixed histories (``feeding_histories``) and their own caution
scale (``feeding.caution_scale``), as ``Life(feeding=True)`` lives them. The diagnosis
does not restart episodes: bodies go on past their episode end (``episode_steps_left``
says where the life would have restarted each one).

The arms (``arms``): how often each joint is within ``ARM_TOL`` of a limit, by energy
band; each drive's weighted reward by posture (folded: 3 or 4 joints at a limit;
partial; extended: none), with caution's share from the joints; whether the joint
action saturates (``|a| > 0.95`` after the tanh, and the policy's mean) and whether it
keeps pushing into a limit it is already at (with the effort lost there); and, sampled
every ``REACH_EVERY`` steps, the fewest steps to put a limb tip on an active food with
the trunk in place (joints only, and joints while turning in place; walls ignored).

``diagnosis.record`` (a list of bodies, or ``zero``: every body that reaches 0) saves
``signals_body<i>.npz`` in the viewer's format (``umbral.world.render.save_gif``, plus
energy, E_int, actions, joints and drives) and ``body<i>.gif`` from 300 steps before its
first step at 0.

It changes nothing and stops nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from umbral.body.body import limb_points_b, strength_b
from umbral.checkpoint import Checkpoint, load_checkpoint
from umbral.config import Config
from umbral.experiments.agent import load_agent
from umbral.experiments.registry import register
from umbral.metrics.behavior import _row_snapshot
from umbral.mind.life import feeding_histories
from umbral.mind.agent import GaussianNoise
from umbral.mind.drives import Drives, hunger_reward, internal_energy_step, temperament_from_config
from umbral.recorder import Recorder
from umbral.world.arena import KIND_NAMES
from umbral.world.batched import BatchedUmbralEnv

BANDS = {"zero": lambda e: e == 0.0, "below_0.05": lambda e: (e > 0) & (e < 0.05), "above_0.2": lambda e: e > 0.2}


def _runs(z: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.r_[False, z, False].astype(int))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


ARM_TOL = 0.1  # rad: a joint this close to one of its limits is "against the limit"
JOINTS = ("left_q1", "left_q2", "right_q1", "right_q2")
ARM_BANDS = {"zero": lambda e: e == 0.0, "below_0.05": lambda e: (e > 0) & (e < 0.05),
             "0.05_0.2": lambda e: (e >= 0.05) & (e <= 0.2), "above_0.2": lambda e: e > 0.2}


def _reach_steps(p, q_now, pos, heading, strength, fpos, fr, grid_q, tips_local, offsets):
    """Fewest steps to put a limb tip on an active food with the trunk in place (joints only, and
    joints while turning in place, in parallel), at the body's strength; walls are ignored.
    Returns (joints_only, with_turning); None when no posture reaches any food."""
    if not len(fpos):
        return None, None
    jstep, tstep = p.max_joint_speed * strength, p.max_turn * strength
    best = [np.inf, np.inf]
    for i in range(2):
        js = np.abs(grid_q - q_now[i]).max(-1) / jstep  # (M,): joints move in parallel
        for k, d in enumerate(offsets):
            c, s = np.cos(heading + d), np.sin(heading + d)
            tl = tips_local[:, i]
            tw = pos + np.stack([c * tl[:, 0] - s * tl[:, 1], s * tl[:, 0] + c * tl[:, 1]], -1)
            gap = np.linalg.norm(tw[:, None] - fpos[None], axis=-1) - fr[None] - p.limb_radius
            ok = (gap <= 0).any(-1)
            if ok.any():
                n = float(np.ceil(max(js[ok].min(), abs(d) / tstep)))  # turning and joints at once
                best[1] = min(best[1], n)
                if d == 0.0:
                    best[0] = min(best[0], float(np.ceil(js[ok].min())))
    return (None if not np.isfinite(best[0]) else best[0]), (None if not np.isfinite(best[1]) else best[1])


def _arms(p, env, energy, q, act, mu, executed, dq, w, gate, value, eaten, pos, heading, food_pos, food_on,
          fatigue, log_std, caution_scale) -> dict[str, Any]:
    """How the arms sit against their limits, what each drive gives there, whether the action saturates,
    and how many steps the body would need to eat the food within its reach (docs/protocolo.md, 26 Nov 2026)."""
    N, T = energy.shape
    lo, hi = p.joint_lo[None, None, None], p.joint_hi[None, None, None]  # (1,1,1,2) over (T,N,limb,joint)
    qq = q[:-1]
    at_lo, at_hi = qq <= lo + ARM_TOL, qq >= hi - ARM_TOL
    at = (at_lo | at_hi).transpose(1, 0, 2, 3).reshape(N, T, 4)  # (N,T,4) in action order
    direction = (at_hi.astype(float) - at_lo.astype(float)).transpose(1, 0, 2, 3).reshape(N, T, 4)
    n_at = at.sum(-1)
    posture = {"folded": n_at >= 3, "partial": (n_at >= 1) & (n_at <= 2), "extended": n_at == 0}
    tips = limb_points_b(p, pos[:-1].reshape(-1, 2), heading[:-1].reshape(-1), qq.reshape(-1, 2, 2))[:, :, 2]
    fwd = ((tips - pos[:-1].reshape(-1, 1, 2)) * np.stack([np.cos(heading[:-1]), np.sin(heading[:-1])], -1)
           .reshape(-1, 1, 2)).sum(-1).max(-1).reshape(T, N).T  # how far ahead of the trunk centre the farther tip is
    aj, mj = act[..., 2:], np.tanh(mu[..., 2:])
    caution_joints = -caution_scale * p.effort_cost * (aj ** 2).sum(-1) / 6.0
    push = (aj * direction > 0.05) & at  # pushing into the limit it is already at
    blocked = push & (np.abs(dq.transpose(1, 0, 2, 3).reshape(N, T, 4)) < 1e-9)
    ex2 = executed ** 2
    out: dict[str, Any] = {"tolerance_rad": ARM_TOL, "joint_lo": p.joint_lo.tolist(), "joint_hi": p.joint_hi.tolist(),
                           "action_std": np.exp(log_std).tolist(), "by_energy": {}, "by_posture": {}}
    for band, sel in ARM_BANDS.items():
        m = sel(energy)
        n = int(m.sum())
        if not n:
            out["by_energy"][band] = {"steps": 0}
            continue
        out["by_energy"][band] = {
            "steps": n, **{f"{k}_fraction": float(v[m].mean()) for k, v in posture.items()},
            "joints_at_limit_mean": float(n_at[m].mean()),
            "at_limit_by_joint": {j: float(at[..., k][m].mean()) for k, j in enumerate(JOINTS)},
            "tip_ahead_of_centre": float(fwd[m].mean())}
    for name, sel in posture.items():
        n = int(sel.sum())
        if not n:
            out["by_posture"][name] = {"steps": 0}
            continue
        f = (lambda a, s=sel: float(a[s].mean()))
        z = sel & (energy == 0.0)
        hi_e = sel & (energy > 0.2)
        out["by_posture"][name] = {
            "steps": n, "energy_mean": f(energy), "at_zero_fraction": float((energy[sel] == 0.0).mean()),
            "hunger": f(w["hunger"]), "curiosity": f(w["curiosity"]), "caution": f(w["caution"]),
            "caution_from_joints": f(caution_joints), "total": f(w["hunger"] + w["curiosity"] + w["caution"]),
            "gate": f(gate), "value": f(value), "eats_per_1000_steps": float(eaten[sel].sum() / n * 1000),
            "tip_ahead_of_centre": f(fwd),
            "above_0.2": None if not hi_e.any() else {
                "steps": int(hi_e.sum()), "hunger": float(w["hunger"][hi_e].mean()),
                "curiosity": float(w["curiosity"][hi_e].mean()), "caution": float(w["caution"][hi_e].mean()),
                "value": float(value[hi_e].mean())},
            "at_zero": None if not z.any() else {
                "steps": int(z.sum()), "hunger": float(w["hunger"][z].mean()),
                "curiosity": float(w["curiosity"][z].mean()), "caution": float(w["caution"][z].mean()),
                "value": float(value[z].mean())},
            "joint_action_abs": float(np.abs(aj)[sel].mean()),
            "joint_action_saturated_0.95": float((np.abs(aj) > 0.95)[sel].mean()),
            "joint_mean_saturated_0.95": float((np.abs(mj) > 0.95)[sel].mean()),
            "joint_mu_abs": float(np.abs(mu[..., 2:])[sel].mean())}
    lim = at
    out["at_limit"] = {
        "joint_steps": int(lim.sum()),
        "pushing_into_limit_fraction": float(push[lim].mean()) if lim.any() else None,
        "mean_toward_limit": float((aj * direction)[lim].mean()) if lim.any() else None,
        "mean_of_policy_toward_limit": float((mj * direction)[lim].mean()) if lim.any() else None,
        "saturated_0.95_fraction": float((np.abs(aj) > 0.95)[lim].mean()) if lim.any() else None,
        "policy_mean_saturated_0.95_fraction": float((np.abs(mj) > 0.95)[lim].mean()) if lim.any() else None,
        "effort_lost_against_limits": float((ex2[..., 2:] * blocked).sum() / max(ex2.sum(), 1e-12)),
        "effort_lost_against_limits_at_zero": float((ex2[..., 2:] * blocked)[energy == 0.0].sum()
                                                    / max(ex2[energy == 0.0].sum(), 1e-12))}
    fold = posture["folded"]
    if fold.any():
        codes, counts = np.unique(direction[fold].astype(int), axis=0, return_counts=True)
        order = np.argsort(-counts)[:5]
        out["by_posture"]["folded"]["patterns"] = [
            {"limits": {j: {1: "hi", -1: "lo", 0: "-"}[int(c)] for j, c in zip(JOINTS, codes[i])},
             "fraction": float(counts[i] / fold.sum())} for i in order]
    out["saturation_all_steps"] = {"joint_action_0.95": float((np.abs(aj) > 0.95).mean()),
                                   "joint_policy_mean_0.95": float((np.abs(mj) > 0.95).mean()),
                                   "trunk_action_0.95": float((np.abs(act[..., :2]) > 0.95).mean())}

    # steps to eat the food within reach: sampled every REACH_EVERY steps per body
    g1 = np.linspace(p.joint_lo[0], p.joint_hi[0], 27)
    g2 = np.linspace(p.joint_lo[1], p.joint_hi[1], 45)
    grid_q = np.stack(np.meshgrid(g1, g2, indexing="ij"), -1).reshape(-1, 2)  # (M,2) one limb
    M = len(grid_q)
    tips_local = limb_points_b(p, np.zeros((M, 2)), np.zeros(M), np.stack([grid_q, grid_q], 1))[:, :, 2]
    offsets = np.r_[0.0, np.linspace(-np.pi, np.pi, 37)[:-1]]
    samples: dict[str, list[dict[str, Any]]] = {k: [] for k in posture}
    for b in range(N):
        for t in range(0, T, REACH_EVERY):
            kind = next(k for k, v in posture.items() if v[b, t])
            fo = food_on[t, b]
            s = float(strength_b(p, energy[b, t:t + 1], fatigue[b, t:t + 1])[0])
            j_only, turn = _reach_steps(p, q[t, b], pos[t, b], heading[t, b], s, food_pos[t, b][fo],
                                        env.food_r[b][fo], grid_q, tips_local, offsets)
            samples[kind].append({"body": b, "t": t, "energy": float(energy[b, t]), "joints_only": j_only,
                                  "with_turning": turn})
    reach: dict[str, Any] = {"sample_every": REACH_EVERY, "walls_ignored": True}
    for k, ss in samples.items():
        if not ss:
            reach[k] = {"samples": 0}
            continue
        jo = np.array([x["joints_only"] for x in ss if x["joints_only"] is not None], float)
        wt = np.array([x["with_turning"] for x in ss if x["with_turning"] is not None], float)
        z = [x for x in ss if x["energy"] == 0.0]
        reach[k] = {"samples": len(ss), "food_within_joints_reach": len(jo) / len(ss),
                    "food_within_reach_turning_in_place": len(wt) / len(ss),
                    "joints_only_steps_median": float(np.median(jo)) if len(jo) else None,
                    "joints_only_steps_max": float(jo.max()) if len(jo) else None,
                    "with_turning_steps_median": float(np.median(wt)) if len(wt) else None,
                    "with_turning_steps_max": float(wt.max()) if len(wt) else None,
                    "at_zero": {"samples": len(z), "within_reach_turning": sum(x["with_turning"] is not None for x in z),
                                "with_turning_steps": sorted(x["with_turning"] for x in z
                                                             if x["with_turning"] is not None)}}
    out["reach_to_eat"] = reach
    return out


REACH_EVERY = 25


@torch.no_grad()
def diagnose(ck: Checkpoint, steps: int, limit: int, debt_floor: float = -1.0,
             record: str | list[int] | None = None) -> dict[str, Any]:
    cfg = Config(ck.config)
    agent, _ = load_agent(ck)
    N = int(cfg.life.n_envs)
    extra = ck.extra or {}
    feeding = extra.get("history") == "all"  # a checkpoint of the common feeding phase: the 4 histories mixed
    histories = feeding_histories(cfg, N) if feeding else [str(cfg.agent.history)] * N
    env = BatchedUmbralEnv(cfg, N, histories)
    ck.restore_env(env)
    p = env.params
    noise = GaussianNoise(N, [0] * N)
    noise.set_state(ck.policy)
    world = agent.world
    h = ck.mind_state["h"].clone()
    prev = ck.mind_state["prev_action"].numpy().astype(float)
    drives = Drives(env.layout, cfg.drives, temperament_from_config(cfg, extra.get("temperament", cfg.agent.temperament)),
                    p.effort_cost)
    caution_scale = (cfg.get("feeding") or {}).get("caution_scale") if feeding else None
    if caution_scale is not None:
        drives.caution_scale = float(caution_scale)  # the childhood's own caution scale, as in Life(feeding=True)
    drives.set_state(extra["drives"])
    D, H, T = env.observation_dim, world.hidden_size, int(steps)
    obs = np.empty((N, T + 1, D))
    act, prv = np.empty((N, T, 6)), np.empty((N, T, 6))
    hid = torch.empty(N, T, H)
    value, energy, effort = np.empty((N, T)), np.empty((N, T)), np.empty((N, T))
    pos, heading = np.empty((T + 1, N, 2)), np.empty((T + 1, N))
    q = np.empty((T + 1, N, 2, 2))
    food_pos, food_on = np.empty((T + 1,) + env.food_pos.shape), np.empty((T + 1,) + env.food_active.shape, bool)
    touch_kind = np.zeros((T, N, 10), int) - 1  # kind touched by each sensor (-1: none)
    eaten = np.zeros((N, T))
    tl = env.layout.slices["touch"]
    o = env.observation()["flat"]
    obs[:, 0] = o
    pos[0], heading[0], q[0] = env.pos, env.heading, env.q
    food_pos[0], food_on[0] = env.food_pos, env.food_active
    e_before = env.energy.copy()
    ei = env.layout.slices["intero"].start
    has_debt = "e_int" in extra
    e_int = np.empty((N, T + 1))
    e_int[:, 0] = np.array(extra["e_int"], float) if has_debt else o[:, ei]
    mu, executed = np.empty((N, T, 6)), np.empty((N, T, 6))
    dq, fatigue = np.empty((T, N, 2, 2)), np.empty((N, T))
    snaps: list[list[dict[str, np.ndarray]]] = [[_row_snapshot(env, r) for r in range(N)]]
    for t in range(T):
        h = world.step_hidden(torch.as_tensor(o, dtype=torch.float32), torch.as_tensor(prev, dtype=torch.float32), h)
        mu[:, t] = agent.policy(h[0])[0].numpy()
        fatigue[:, t] = env.fatigue
        a, _, _, v = agent.act(h[0], noise.sample())
        res, _, _, _, info = env.step(a)
        executed[:, t], dq[t] = info["action_executed"], env.dq
        snaps.append([_row_snapshot(env, r) for r in range(N)])
        o = res["flat"]
        hid[:, t], value[:, t] = h[0], v
        prv[:, t], act[:, t] = prev, info["action_intended"]
        prev = info["action_intended"]
        obs[:, t + 1] = o
        energy[:, t] = e_before
        e_before = info["energy"]
        effort[:, t] = (info["action_executed"] ** 2).mean(-1)
        eaten[:, t] = info["food_eaten"]
        bal = info["energy_balance"]
        cost = np.zeros(N) if bal is None else np.where(info["frozen"], 0.0, bal["basal"] + bal["effort_cost"])
        e_int[:, t + 1] = internal_energy_step(e_int[:, t], o[:, ei], cost, info["food_eaten"] > 0, debt_floor)
        pos[t + 1], heading[t + 1], q[t + 1] = env.pos, env.heading, env.q
        food_pos[t + 1], food_on[t + 1] = env.food_pos, env.food_active
        tt = env.raw[:, tl].reshape(N, 10, 6)
        touch_kind[t] = np.where(tt[..., 0] > 0, tt[..., 1:].argmax(-1), -1)
    cur, nxt = torch.as_tensor(obs[:, :-1], dtype=torch.float32), torch.as_tensor(obs[:, 1:], dtype=torch.float32)
    pred = world.predict_next(cur, world.from_hidden(cur, hid, torch.as_tensor(act, dtype=torch.float32)))
    _, comps = drives(obs[:, :-1], obs[:, 1:], pred.double().numpy(), act, e_int)
    w = comps["weighted"]
    sp, hs = drives.set_point, drives.hunger_scale
    h_v1 = hunger_reward(obs[:, :-1, ei], obs[:, 1:, ei], sp, hs)
    h_debt = hunger_reward(e_int[:, :-1], e_int[:, 1:], sp, hs)

    out: dict[str, Any] = {"checkpoint_step": ck.step, "temperament": drives.temperament.name,
                           "history": "all (feeding phase)" if feeding else str(cfg.agent.history),
                           "histories_of_bodies": histories, "caution_scale": float(drives.caution_scale),
                           "metabolism": float(env.metabolism),
                           "episode_steps_left": [int(x) * int(cfg.life.rollout) for x in extra.get("life_left", [])],
                           "episode_resets": "none in the diagnosis (bodies go on past their episode end)",
                           "steps": T, "bodies": N, "hunger_debt_in_config": drives.hunger_debt,
                           "debt_start": "checkpoint" if has_debt else "observed energy (no debt record)",
                           "debt_floor": float(debt_floor), "debt_min": float(e_int.min()),
                           "debt_floor_steps": int((e_int[:, 1:] <= debt_floor).sum()), "bands": {}}
    for band, sel in BANDS.items():
        m = sel(energy)
        n = int(m.sum())
        f = (lambda a: float(a[m].mean()) if n else None)
        out["bands"][band] = {
            "steps": n, "hunger": f(w["hunger"]), "curiosity": f(w["curiosity"]), "caution": f(w["caution"]),
            "total": f(w["hunger"] + w["curiosity"] + w["caution"]), "gate": f(comps["gate"]), "value": f(value),
            "hunger_v1": f(h_v1), "hunger_debt": f(h_debt), "e_int_mean": f(e_int[:, :-1]),
            "action_abs": float(np.abs(act)[m].mean()) if n else None, "effort": f(effort),
            "eats_per_1000_steps": float(eaten[m].sum() / n * 1000) if n else None,
            "hunger_positive_fraction": float((w["hunger"][m] > 0).mean()) if n else None}

    eth = extra.get("ethics", {})
    seen: dict[str, Any] = {}
    for ph in ("infancy", "adult"):
        hist = np.asarray(eth.get(f"hist_{ph}", []), float)
        if hist.size and hist.sum():
            seen[ph] = {"steps": int(hist.sum()), "below_0.01": float(hist[0] / hist.sum()),
                        "below_0.05": float(hist[:5].sum() / hist.sum()), "below_0.2": float(hist[:20].sum() / hist.sum())}
    out["training_exposure"] = seen

    s0 = float(strength_b(p, np.zeros(1), np.zeros(1))[0])
    speed, turn = p.max_speed * s0, p.max_turn * s0
    bodies = []
    for b in range(N):
        for s, e in _runs(energy[b] == 0.0):
            if e - s < limit:
                continue
            fo = food_on[s, b]
            c = pos[s, b]
            if not fo.any():
                bodies.append({"body": b, "history": histories[b], "start": s, "length": e - s, "active_food": 0})
                continue
            d = np.linalg.norm(food_pos[s, b] - c, axis=-1)
            d = np.where(fo, d, np.inf)
            j = int(d.argmin())
            fr = float(env.food_r[b, j])
            ang = abs(float(_angle(np.arctan2(*(food_pos[s, b, j] - c)[::-1]) - heading[s, b])))
            walk = max(0.0, float(d[j]) - p.trunk_radius - p.reach - fr)  # until a tip at full stretch reaches it
            need = ang / turn + walk / speed
            d_end = float(np.linalg.norm(food_pos[s, b, j] - pos[e, b]))
            path = float(np.linalg.norm(np.diff(pos[s:e + 1, b], axis=0), axis=-1).sum())
            turned = float(np.abs(_angle(np.diff(heading[s:e + 1, b]))).sum())
            bodies.append({"body": b, "history": histories[b], "start": s, "length": e - s, "ends_by": "window end" if e == T else "energy rose",
                           "nearest_food": float(d[j]), "turn_needed_rad": ang, "straight_line_steps_needed": need,
                           "strength_at_zero": s0, "path": path, "net_approach": float(d[j]) - d_end,
                           "turned_rad": turned,
                           "hunger_before": float(h_v1[b, s - 1]) if s > 0 else None,
                           "hunger_v1_during": float(h_v1[b, s:e].mean()),
                           "hunger_debt_during": float(h_debt[b, s:e].mean()),
                           "hunger_debt_first": float(h_debt[b, s]), "hunger_debt_last": float(h_debt[b, e - 1]),
                           "e_int_end": float(e_int[b, e]), "box": [float(np.ptp(pos[s:e + 1, b, 0])), float(np.ptp(pos[s:e + 1, b, 1]))],
                           "contacts": {KIND_NAMES[k]: float((touch_kind[s:e, b] == k).sum(-1).mean())
                                        for k in range(5) if (touch_kind[s:e, b] == k).any()}})
    out["zero_stretches"] = bodies

    # touching food without eating: food on a sensor for >= 100 steps in a row with no food eaten
    lr = p.limb_radius
    grid = np.linspace(0, 1, 15)
    touching = []
    for b in range(N):
        food_touch = (touch_kind[:, b] == 1).any(-1)
        for s, e in _runs(food_touch & (eaten[b] == 0)):
            if e - s < 100:
                continue
            sensors = sorted({int(k) for k in np.flatnonzero((touch_kind[s:e, b] == 1).any(0))})
            fo = food_on[s + 1, b]
            fp, fr = food_pos[s + 1, b][fo], env.food_r[b][fo]
            tips = limb_points_b(p, pos[s + 1, b][None], heading[s + 1, b:b + 1], q[s + 1, b][None])[0, :, 2]
            gaps = np.linalg.norm(fp[None] - tips[:, None], axis=-1) - fr[None] - lr  # (2 limbs, foods)
            k = int(gaps.min(0).argmin())
            # can the joints alone (trunk still) bring a tip onto that food?
            best = np.inf
            for i1 in grid:
                for i2 in grid:
                    qq = np.repeat(q[s + 1, b][None], 2, 0)
                    a1 = p.joint_lo + (p.joint_hi - p.joint_lo) * np.array([i1, i2])
                    qq[0, 0], qq[1, 1] = a1, a1
                    tp = limb_points_b(p, np.repeat(pos[s + 1, b][None], 2, 0), np.repeat(heading[s + 1, b], 2),
                                       qq)[:, :, 2]
                    g = np.linalg.norm(tp - fp[k], axis=-1) - fr[k] - lr
                    best = min(best, float(g.min()))
            touching.append({"body": b, "start": s, "length": e - s, "sensors": sensors,
                             "sensor_parts": ["contour" if x < 8 else "tip" for x in sensors],
                             "tip_gaps": [float(x) for x in gaps[:, k]], "trunk_to_food": float(
                                 np.linalg.norm(fp[k] - pos[s + 1, b]) - p.trunk_radius - fr[k]),
                             "joints": q[s + 1, b].tolist(), "joint_lo": list(map(float, p.joint_lo)),
                             "joint_hi": list(map(float, p.joint_hi)), "joints_alone_best_gap": best,
                             "joint_actions_mean": act[b, s:e, 2:].mean(0).tolist(),
                             "moved": float(np.linalg.norm(pos[e, b] - pos[s, b]))})
    out["touching_food_without_eating"] = touching
    out["arms"] = _arms(p, env, energy, q, act, mu, executed, dq, w, comps["gate"], value, eaten, pos, heading,
                        food_pos, food_on, fatigue, agent.policy.log_std.detach().numpy(), float(drives.caution_scale))
    out["bodies_reaching_zero"] = [int(b) for b in np.flatnonzero((energy == 0.0).any(-1))]
    out["per_body"] = [{"body": b, "history": histories[b], "energy_min": float(energy[b].min()),
                        "steps_at_zero": int((energy[b] == 0.0).sum()), "eaten": int(eaten[b].sum()),
                        "e_int_min": float(e_int[b].min())} for b in range(N)]
    rec = {}
    for b in out["bodies_reaching_zero"] if record == "zero" else list(record or []):
        b = int(b)
        arena, r = env.row_arena(b), {}
        for k in snaps[0][b]:
            r[k] = np.stack([sn[b][k] for sn in snaps[1:]]).astype(bool if k == "food_active" else np.float32)
        r.update(energy=energy[b].astype(np.float32), e_int=e_int[b, :-1].astype(np.float32),
                 action_intended=act[b].astype(np.float32), action_mean=np.tanh(mu[b]).astype(np.float32),
                 action_executed=executed[b].astype(np.float32), joints=q[:-1, b].astype(np.float32),
                 hunger=w["hunger"][b].astype(np.float32), curiosity=w["curiosity"][b].astype(np.float32),
                 caution=w["caution"][b].astype(np.float32), value=value[b].astype(np.float32),
                 food_eaten=eaten[b].astype(np.int64), arena_size=np.float32(arena.size),
                 trunk_radius=np.float32(p.trunk_radius), obstacle_pos=arena.obstacle_pos.astype(np.float32),
                 obstacle_r=arena.obstacle_r.astype(np.float32), food_r=arena.food_r.astype(np.float32),
                 neutral_r=np.r_[env.neutral_r[b, :int(env.neutral_on[b, :env.probe_slot].sum())],
                                 float(cfg.world.neutral_radius)].astype(np.float32),
                 history_id=np.array(histories[b]), body=np.int64(b))
        rec[b] = r
    out["_recordings"] = rec
    return out


@register("zero_diagnosis")
def run_zero_diagnosis(config: Config, recorder: Recorder, np_rng: np.random.Generator,
                       resume: Checkpoint | None) -> None:
    from umbral.world.render import save_gif

    path = Path(str(config.diagnosis.checkpoint))
    record = config.diagnosis.record
    record = None if record is None else (str(record) if isinstance(record, str) else [int(x) for x in record])
    out = {"checkpoint": str(path), **diagnose(load_checkpoint(path), int(config.diagnosis.steps),
                                                int(config.ethics.zero_run_limit or 200),
                                                float(config.drives.debt_floor), record)}
    recs = out.pop("_recordings")
    out["recordings"] = {}
    for b, sig in recs.items():
        np.savez_compressed(recorder.run_dir / f"signals_body{b}.npz", **sig)
        z = np.flatnonzero(sig["energy"] == 0.0)
        start = max(0, int(z[0]) - 300) if len(z) else 0
        n = min(1200, len(sig["energy"]) - start)
        clip = {k: (v[start:] if getattr(v, "ndim", 0) and len(v) == len(sig["energy"]) else v) for k, v in sig.items()}
        save_gif(clip, recorder.run_dir / f"body{b}.gif", steps=n, stride=6, fps=15,
                 title=f"h{sig['history_id']} body {b} · from step {start} ·")
        out["recordings"][b] = {"signals": f"signals_body{b}.npz", "gif": f"body{b}.gif", "gif_from_step": start,
                                "gif_steps": n}
    (recorder.run_dir / "diagnosis.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for band, v in out["bands"].items():
        recorder.log(f"{band}: " + "  ".join(f"{k} {x:.4g}" if isinstance(x, float) else f"{k} {x}"
                                             for k, x in v.items()))
    recorder.log(f"training exposure: {out['training_exposure']}")
    recorder.log(f"stretches at energy 0 >= limit: {len(out['zero_stretches'])}; "
                 f"touching food without eating: {len(out['touching_food_without_eating'])}")
    recorder.update_meta(agent_checkpoint=str(path), agent_step=out["checkpoint_step"])
