"""``random_agent``: a body without a mind, driven by smoothed random actions.

Actions follow an Ornstein-Uhlenbeck-like process drawn from the run's numpy
Generator: ``a <- s * a + sqrt(1 - s^2) * scale * N(0, 1)``, clipped to [-1, 1]
by the environment. The BodyLink stays ``intact``.

Outputs in the run directory:
- metrics per step: ``energy``, ``food_eaten`` (cumulative), ``contacts``
  (active touch sensors) and ``void_action`` (sum over parts; 0 while intact);
- ``signals.npz``: every signal per step (see ``SIGNAL_KEYS``) plus static arena data;
- ``run.gif``: the first ``gif_steps`` steps;
- ``meta.json``: ``env_steps_per_second`` (time inside ``env.step`` only) and
  ``loop_steps_per_second`` (whole loop including recording).
"""

from __future__ import annotations

import time

import numpy as np

from umbral.checkpoint import Checkpoint
from umbral.config import Config
from umbral.experiments.registry import register
from umbral.recorder import Recorder
from umbral.world.env import UmbralEnv
from umbral.world.render import save_gif

SIGNAL_KEYS = (
    "obs_flat", "raw_flat", "proprio", "touch", "vision", "intero",
    "action_intended", "action_applied", "action_executed", "void_action",
    "energy", "fatigue", "food_eaten", "contacts", "frozen",
    "trunk_pose", "limb_points", "food_pos", "food_active", "neutral_pos", "touch_intensity",
)


@register("random_agent")
def run_random_agent(config: Config, recorder: Recorder, np_rng: np.random.Generator,
                     resume: Checkpoint | None) -> None:
    if resume is not None:
        raise ValueError("random_agent does not support resuming from a checkpoint")
    c = config.random_agent
    env = UmbralEnv(config)
    env_seed = int(np_rng.integers(0, 2**31 - 1))
    obs, info = env.reset(seed=env_seed)
    recorder.log(f"history {info['history_id']} ({info['history_name']}), env seed {env_seed}, "
                 f"obs dim {env.observation_dim}")

    rec: dict[str, list] = {k: [] for k in SIGNAL_KEYS}
    a = np.zeros(env.action_dim)
    s, scale = float(c.smoothing), float(c.scale)
    noise_gain = np.sqrt(1.0 - s * s) * scale
    env_time = 0.0
    t_loop = time.perf_counter()
    for step in range(1, int(c.steps) + 1):
        a = s * a + noise_gain * np_rng.standard_normal(env.action_dim)
        t0 = time.perf_counter()
        obs, _, _, _, info = env.step(a)
        env_time += time.perf_counter() - t0

        void = sum(info["void_action"].values())
        for key in ("proprio", "touch", "vision", "intero"):
            rec[key].append(obs[key])
        rec["obs_flat"].append(obs["flat"])
        rec["raw_flat"].append(env._raw.copy())
        for key in ("action_intended", "action_applied", "action_executed", "energy", "fatigue",
                    "food_eaten", "contacts", "frozen"):
            rec[key].append(info[key])
        rec["void_action"].append([info["void_action"][p] for p in ("trunk", "left_limb", "right_limb")])
        for key, val in env.snapshot().items():
            rec[key].append(val)

        recorder.log_metric(step, "energy", info["energy"])
        recorder.log_metric(step, "food_eaten", env.food_eaten_total)
        recorder.log_metric(step, "contacts", info["contacts"])
        recorder.log_metric(step, "void_action", void)
        if step % 500 == 0:
            recorder.log(f"step {step}: energy={info['energy']:.3f} food={env.food_eaten_total}")
    loop_time = time.perf_counter() - t_loop
    n = int(c.steps)

    arrays = {k: np.asarray(v, dtype=np.float32 if k not in ("food_active", "frozen", "food_eaten", "contacts")
                            else None) for k, v in rec.items()}
    p, arena = env.params, env.arena
    arrays.update(
        arena_size=np.float32(arena.size), trunk_radius=np.float32(p.trunk_radius),
        obstacle_pos=arena.obstacle_pos.astype(np.float32), obstacle_r=arena.obstacle_r.astype(np.float32),
        food_r=arena.food_r.astype(np.float32), neutral_r=arena.neutral_r.astype(np.float32),
        history_id=np.array(env.history.id), env_seed=np.int64(env_seed),
    )
    npz_path = recorder.run_dir / "signals.npz"
    np.savez_compressed(npz_path, **arrays)
    recorder.log(f"signals saved: {npz_path}")

    gif_path = save_gif(arrays, recorder.run_dir / "run.gif", steps=int(c.gif_steps),
                        stride=int(c.gif_stride), fps=int(c.gif_fps),
                        title=f"{env.history.name} ·")
    recorder.log(f"gif saved: {gif_path}")

    recorder.update_meta(
        env_seed=env_seed, history_id=env.history.id, history_name=env.history.name,
        observation_dim=env.observation_dim, action_dim=env.action_dim,
        env_steps_per_second=round(n / env_time, 1),
        loop_steps_per_second=round(n / loop_time, 1),
        food_eaten_total=env.food_eaten_total, final_energy=env.body.energy,
    )
    recorder.log(f"env steps/s: {n / env_time:.0f} (loop incl. recording: {n / loop_time:.0f})")
