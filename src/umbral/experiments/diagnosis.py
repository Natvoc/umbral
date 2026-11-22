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
from umbral.mind.agent import GaussianNoise
from umbral.mind.drives import Drives, temperament_from_config
from umbral.recorder import Recorder
from umbral.world.arena import KIND_NAMES
from umbral.world.batched import BatchedUmbralEnv

BANDS = {"zero": lambda e: e == 0.0, "below_0.05": lambda e: (e > 0) & (e < 0.05), "above_0.2": lambda e: e > 0.2}


def _runs(z: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.r_[False, z, False].astype(int))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


@torch.no_grad()
def diagnose(ck: Checkpoint, steps: int, limit: int) -> dict[str, Any]:
    cfg = Config(ck.config)
    agent, _ = load_agent(ck)
    N = int(cfg.life.n_envs)
    env = BatchedUmbralEnv(cfg, N, [str(cfg.agent.history)] * N)
    ck.restore_env(env)
    p = env.params
    noise = GaussianNoise(N, [0] * N)
    noise.set_state(ck.policy)
    world = agent.world
    h = ck.mind_state["h"].clone()
    prev = ck.mind_state["prev_action"].numpy().astype(float)
    extra = ck.extra or {}
    drives = Drives(env.layout, cfg.drives, temperament_from_config(cfg, extra.get("temperament", cfg.agent.temperament)),
                    p.effort_cost)
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
    for t in range(T):
        h = world.step_hidden(torch.as_tensor(o, dtype=torch.float32), torch.as_tensor(prev, dtype=torch.float32), h)
        a, _, _, v = agent.act(h[0], noise.sample())
        res, _, _, _, info = env.step(a)
        o = res["flat"]
        hid[:, t], value[:, t] = h[0], v
        prv[:, t], act[:, t] = prev, info["action_intended"]
        prev = info["action_intended"]
        obs[:, t + 1] = o
        energy[:, t] = e_before
        e_before = info["energy"]
        effort[:, t] = (info["action_executed"] ** 2).mean(-1)
        eaten[:, t] = info["food_eaten"]
        pos[t + 1], heading[t + 1], q[t + 1] = env.pos, env.heading, env.q
        food_pos[t + 1], food_on[t + 1] = env.food_pos, env.food_active
        tt = env.raw[:, tl].reshape(N, 10, 6)
        touch_kind[t] = np.where(tt[..., 0] > 0, tt[..., 1:].argmax(-1), -1)
    cur, nxt = torch.as_tensor(obs[:, :-1], dtype=torch.float32), torch.as_tensor(obs[:, 1:], dtype=torch.float32)
    pred = world.predict_next(cur, world.from_hidden(cur, hid, torch.as_tensor(act, dtype=torch.float32)))
    _, comps = drives(obs[:, :-1], obs[:, 1:], pred.double().numpy(), act)
    w = comps["weighted"]

    out: dict[str, Any] = {"checkpoint_step": ck.step, "temperament": drives.temperament.name,
                           "history": str(cfg.agent.history), "metabolism": float(env.metabolism),
                           "steps": T, "bodies": N, "bands": {}}
    for band, sel in BANDS.items():
        m = sel(energy)
        n = int(m.sum())
        f = (lambda a: float(a[m].mean()) if n else None)
        out["bands"][band] = {
            "steps": n, "hunger": f(w["hunger"]), "curiosity": f(w["curiosity"]), "caution": f(w["caution"]),
            "total": f(w["hunger"] + w["curiosity"] + w["caution"]), "gate": f(comps["gate"]), "value": f(value),
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
                bodies.append({"body": b, "start": s, "length": e - s, "active_food": 0})
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
            bodies.append({"body": b, "start": s, "length": e - s, "ends_by": "window end" if e == T else "energy rose",
                           "nearest_food": float(d[j]), "turn_needed_rad": ang, "straight_line_steps_needed": need,
                           "strength_at_zero": s0, "path": path, "net_approach": float(d[j]) - d_end,
                           "turned_rad": turned, "box": [float(np.ptp(pos[s:e + 1, b, 0])), float(np.ptp(pos[s:e + 1, b, 1]))],
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
    return out


@register("zero_diagnosis")
def run_zero_diagnosis(config: Config, recorder: Recorder, np_rng: np.random.Generator,
                       resume: Checkpoint | None) -> None:
    path = Path(str(config.diagnosis.checkpoint))
    out = {"checkpoint": str(path), **diagnose(load_checkpoint(path), int(config.diagnosis.steps),
                                                int(config.ethics.zero_run_limit or 200))}
    (recorder.run_dir / "diagnosis.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for band, v in out["bands"].items():
        recorder.log(f"{band}: " + "  ".join(f"{k} {x:.4g}" if isinstance(x, float) else f"{k} {x}"
                                             for k, x in v.items()))
    recorder.log(f"training exposure: {out['training_exposure']}")
    recorder.log(f"stretches at energy 0 >= limit: {len(out['zero_stretches'])}; "
                 f"touching food without eating: {len(out['touching_food_without_eating'])}")
    recorder.update_meta(agent_checkpoint=str(path), agent_step=out["checkpoint_step"])
