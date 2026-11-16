"""Training of the predictive mind (Phase 2a) and shared helpers.

Two models are trained side by side on exactly the same data, in the same order:
``efference`` (the mind) and ``no_efference`` (control: same architecture, same
initial weights, actions replaced by zeros). They share nothing else: separate
parameters and optimizers, no shared gradients. Neither influences the policy,
which is the smoothed random OU policy of Phase 1.

Seeds (disjoint ranges):
- training environments: drawn from the run's numpy Generator in [0, 2**31);
- evaluation environments: ``EVAL_SEED_BASE + i`` (>= 2**31); the light evaluation
  at every checkpoint uses ``CKPT_EVAL_SEED_BASE + i`` (fixed, also >= 2**31);
- normalization warm-up: ``NORM_SEED + i`` (>= 2**32), run once before training.

Truncated backpropagation through time: segments of ``train.seq_len`` steps,
hidden states carried (detached) between segments and reset when an
environment starts a new episode (every ``train.episode_steps`` steps, staggered
across environments).
"""

from __future__ import annotations

import copy
import ctypes
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from umbral.checkpoint import Checkpoint, make_checkpoint, save_checkpoint
from umbral.config import Config
from umbral.mind.model import CHANNELS, WorldModel, build_model, channel_sq_error, n_parameters
from umbral.mind.policy import OUPolicy
from umbral.recorder import Recorder
from umbral.world.batched import BatchedUmbralEnv

TRAIN_SEED_LIMIT = 2**31
EVAL_SEED_BASE = 2**31
CKPT_EVAL_SEED_BASE = 2**31 + 500_000  # light evaluation at checkpoints (disjoint from the final eval)
NORM_SEED = 2**32 + 12345
POLICY_SEED_OFFSET = 1_000_000
MODEL_NAMES = ("efference", "no_efference")


def history_cycle(config: Config, n: int) -> list[str]:
    hids = sorted(config.world.histories)
    return [hids[i % len(hids)] for i in range(n)]


def peak_memory_mb() -> float | None:
    """Peak resident memory of this process in MB (Windows and Unix), or None."""
    try:
        if sys.platform == "win32":
            class PMC(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            k32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return pmc.PeakWorkingSetSize / 2**20
            return None
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return None


def estimate_delta_std(config: Config, model: WorldModel) -> np.ndarray:
    """Per-element std of one-step changes of the regression targets, from a fixed-seed warm-up.

    Runs once, with seeds outside the training and evaluation ranges, and is then
    frozen in the models' state_dict.
    """
    tc = config.train
    n = int(tc.norm_envs)
    env = BatchedUmbralEnv(config, n, history_cycle(config, n))
    env.reset(seed=[NORM_SEED + i for i in range(n)])
    policy = OUPolicy(n, float(tc.smoothing), float(tc.scale), [NORM_SEED + POLICY_SEED_OFFSET + i for i in range(n)])
    idx = model.out_layout.reg_idx
    prev = env.observation()["flat"][:, idx]
    deltas = np.empty((int(tc.norm_steps), n, len(idx)))
    for t in range(int(tc.norm_steps)):
        obs, *_ = env.step(policy.act())
        cur = obs["flat"][:, idx]
        deltas[t] = cur - prev
        prev = cur
    return np.maximum(deltas.reshape(-1, len(idx)).std(axis=0), 1e-6)


class Trainer:
    def __init__(self, config: Config, recorder: Recorder, np_rng: np.random.Generator, resume: Checkpoint | None):
        self.config, self.rec = config, recorder
        tc = config.train
        self.N, self.T = int(tc.n_envs), int(tc.seq_len)
        self.updates = int(tc.updates)
        self.log_every, self.ckpt_every = int(tc.log_every), int(tc.checkpoint_every)
        if int(tc.episode_steps) % self.T:
            raise ValueError("train.episode_steps must be a multiple of train.seq_len")
        if self.ckpt_every % self.log_every:
            raise ValueError("train.checkpoint_every must be a multiple of train.log_every")
        self.episode_segments = int(tc.episode_steps) // self.T
        self.histories = history_cycle(config, self.N)
        self.env = BatchedUmbralEnv(config, self.N, self.histories)
        layout = self.env.layout

        # Same architecture and the same initial weights; only the efference copy differs.
        eff = build_model(layout, config.mind, efference_copy=True)
        noeff = build_model(layout, config.mind, efference_copy=False)
        noeff.load_state_dict(eff.state_dict())
        self.models: dict[str, WorldModel] = {"efference": eff, "no_efference": noeff}
        self.opts = {k: torch.optim.Adam(m.parameters(), lr=float(tc.lr)) for k, m in self.models.items()}

        if resume is None:
            self.np_rng = np_rng
            std = estimate_delta_std(config, eff)
            for m in self.models.values():
                m.set_delta_std(std)
            pol_seeds = [int(np_rng.integers(0, TRAIN_SEED_LIMIT)) for _ in range(self.N)]
            self.policy = OUPolicy(self.N, float(tc.smoothing), float(tc.scale), pol_seeds)
            self.env.reset(seed=[self._new_seed() for _ in range(self.N)])
            self.h = {k: m.initial_state(self.N) for k, m in self.models.items()}
            self.prev_action = np.zeros((self.N, 6))
            E = self.episode_segments
            self.life_left = np.array([E - (i * E) // self.N for i in range(self.N)])
            self.start = 0
            self.elapsed_before = 0.0
        else:
            resume.load_into(self.models, self.opts)
            self.policy = OUPolicy(self.N, float(tc.smoothing), float(tc.scale), [0] * self.N)
            self.policy.set_state(copy.deepcopy(resume.policy))
            resume.restore_env(self.env)
            ms = resume.mind_state
            self.h = {k: ms[f"h_{k}"].clone() for k in self.models}
            self.prev_action = ms["prev_action"].numpy().astype(float).copy()
            self.life_left = np.array(resume.extra["life_left"], dtype=int)
            self.start = int(resume.step)
            self.elapsed_before = float(resume.extra.get("elapsed_seconds", 0.0))
            self.np_rng = resume.activate()  # after building models: restores global and seed-stream RNGs
        self.times = {"world": 0.0, **{f"model_{k}": 0.0 for k in self.models}, "metrics_and_io": 0.0}
        self._reset_acc()
        recorder.update_meta(
            n_parameters=n_parameters(eff), histories=self.histories, n_envs=self.N, seq_len=self.T,
            delta_std=[round(float(x), 8) for x in eff.delta_std])

    def _new_seed(self) -> int:
        return int(self.np_rng.integers(0, TRAIN_SEED_LIMIT))

    def _reset_acc(self) -> None:
        self.acc_loss = {k: {c: 0.0 for c in CHANNELS} for k in self.models}
        self.acc_sq = {k: {c: [0.0, 0] for c in CHANNELS} for k in (*self.models, "persistence")}
        self.acc_n = 0

    # --- one update ----------------------------------------------------------------

    def _collect(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = np.flatnonzero(self.life_left == 0)
        if len(rows):
            self.env.reset(seed=[self._new_seed() for _ in rows], envs=rows)
            self.policy.reset_rows(rows)
            for k in self.h:
                self.h[k][:, rows] = 0.0
            self.prev_action[rows] = 0.0
            self.life_left[rows] = self.episode_segments
        N, T, D = self.N, self.T, self.env.observation_dim
        obs = np.empty((N, T + 1, D))
        act = np.empty((N, T, 6))
        prev = np.empty((N, T, 6))
        obs[:, 0] = self.env.observation()["flat"]
        for t in range(T):
            a = self.policy.act()
            o, _, _, _, info = self.env.step(a)
            intended = info["action_intended"]  # the efference copy is the (clipped) intended action
            prev[:, t] = self.prev_action
            act[:, t] = intended
            obs[:, t + 1] = o["flat"]
            self.prev_action = intended
        self.life_left -= 1
        return obs, prev, act

    def step(self, u: int) -> None:
        t0 = time.perf_counter()
        obs_np, prev_np, act_np = self._collect()
        t1 = time.perf_counter()
        self.times["world"] += t1 - t0
        obs = torch.as_tensor(obs_np, dtype=torch.float32)
        cur, nxt = obs[:, :-1], obs[:, 1:]
        prev = torch.as_tensor(prev_np, dtype=torch.float32)
        act = torch.as_tensor(act_np, dtype=torch.float32)
        layout = self.env.layout
        clip = float(self.config.train.grad_clip)
        for k, model in self.models.items():
            ts = time.perf_counter()
            out = model(cur, prev, act, self.h[k])
            losses = model.loss(cur, nxt, out)
            self.opts[k].zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            self.opts[k].step()
            self.h[k] = out.h_last.detach()
            with torch.no_grad():
                sq = channel_sq_error(layout, model.predict_next(cur, out), nxt)
            for c in CHANNELS:
                self.acc_loss[k][c] += float(losses[c].detach())
                self.acc_sq[k][c][0] += sq[c][0]
                self.acc_sq[k][c][1] += sq[c][1]
            self.times[f"model_{k}"] += time.perf_counter() - ts
        ts = time.perf_counter()
        for c, (s, n) in channel_sq_error(layout, cur, nxt).items():
            self.acc_sq["persistence"][c][0] += s
            self.acc_sq["persistence"][c][1] += n
        self.acc_n += 1
        if u % self.log_every == 0:
            for k in self.models:
                for c in CHANNELS:
                    self.rec.log_metric(u, f"{k}/loss_{c}", self.acc_loss[k][c] / self.acc_n)
            for k, per in self.acc_sq.items():
                for c in CHANNELS:
                    self.rec.log_metric(u, f"{k}/mse_{c}", per[c][0] / per[c][1])
            self._reset_acc()
        self.times["metrics_and_io"] += time.perf_counter() - ts

    # --- loop ------------------------------------------------------------------------

    def checkpoint(self, u: int, elapsed: float) -> Path:
        ckpt = make_checkpoint(
            u, self.models, self.opts, self.np_rng, self.config,
            extra={"life_left": self.life_left.tolist(), "histories": self.histories, "elapsed_seconds": elapsed},
            env=self.env, policy=self.policy,
            mind_state={**{f"h_{k}": v for k, v in self.h.items()},
                        "prev_action": torch.as_tensor(self.prev_action)})
        return save_checkpoint(self.rec.run_dir / "checkpoints" / f"update_{u:07d}.pt", ckpt)

    def checkpoint_eval(self, u: int) -> None:
        """Light evaluation on fixed seeds (the same at every checkpoint), logged as ``ckpt_eval/...``."""
        from umbral.mind.evaluation import TOUCH_EVENTS, evaluate, summarize

        tc = self.config.train
        hids = sorted(self.config.world.histories)
        histories = [h for h in hids for _ in range(int(tc.ckpt_eval_envs_per_history))]
        n = len(histories)
        sums = evaluate(self.models, self.config, histories,
                        [CKPT_EVAL_SEED_BASE + i for i in range(n)],
                        [CKPT_EVAL_SEED_BASE + POLICY_SEED_OFFSET + i for i in range(n)],
                        int(tc.ckpt_eval_steps), 64)
        res = summarize(sums)
        for c in CHANNELS:
            for p, v in res["mse"][c].items():
                self.rec.log_metric(u, f"ckpt_eval/{p}/mse_{c}", v)
        for e in TOUCH_EVENTS:
            for p, v in res["touch_events"][e].items():
                if v is not None:
                    self.rec.log_metric(u, f"ckpt_eval/{p}/touch_{e}", v)

    def run(self) -> dict[str, Any]:
        wall0 = time.perf_counter()
        done_here = 0
        for u in range(self.start + 1, self.updates + 1):
            self.step(u)
            done_here += 1
            elapsed = time.perf_counter() - wall0
            if u % self.ckpt_every == 0 or u == self.updates:
                ts = time.perf_counter()
                path = self.checkpoint(u, self.elapsed_before + elapsed)
                self.checkpoint_eval(u)
                self.times["metrics_and_io"] += time.perf_counter() - ts
                self.rec.log(f"checkpoint: {path.name}")
            if u % self.log_every == 0 or u == self.updates:
                per = elapsed / done_here
                eta = per * (self.updates - u)
                self.rec.log(f"update {u}/{self.updates}  {per * 1e3:.0f} ms/update  "
                             f"elapsed {elapsed / 60:.1f} min  ETA {eta / 60:.1f} min")
        wall = time.perf_counter() - wall0
        total = sum(self.times.values()) or 1.0
        summary = {
            "updates_this_run": done_here,
            "wall_seconds": round(wall, 1),
            "seconds_per_update": round(wall / max(done_here, 1), 4),
            "env_steps_per_second_in_training": round(done_here * self.N * self.T / max(wall, 1e-9), 1),
            "time_fractions": {k: round(v / total, 4) for k, v in self.times.items()},
            "peak_memory_mb": None if peak_memory_mb() is None else round(peak_memory_mb(), 1),
        }
        self.rec.update_meta(**summary)
        return summary
