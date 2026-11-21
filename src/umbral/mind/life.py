"""The agent's own life (Phase 2b): it acts by its drives and keeps learning its body.

Common childhood, in two parts, then the own life:

1. *Learning the body*: the Phase 2a predictive model (efference copy, frozen
   normalization; ``agent.infancy``).
2. *Learning to eat* (``Life(feeding=True)``, experiment ``common_feeding``): one
   shared phase, the same for every agent. It starts from part 1 with a new actor and
   critic, and hunger is its only drive (curiosity and caution weigh 0; the curiosity
   running mean is still tracked, so it reaches the lives calibrated). The bodies are
   spread over the 4 histories (``life.n_envs / 4`` each), so eating does not depend
   on one world. The self-model keeps learning with the prediction loss only. The
   same protected childhood ramp and ethics rule as a life apply. Environment seeds
   come from their own range (``FEEDING_SEED_BASE + [0, 2**31)``). Every
   ``feeding.eval_every`` iterations from the end of the ramp, ``feeding_eval`` runs
   the termination criterion: full cost, stochastic policy, fixed evaluation seeds
   (``FEEDING_EVAL_SEED_BASE + i``), every history on its own. When all histories meet
   it, ``checkpoints/common_start.pt`` is saved: the single starting point of every
   life. If the time cap ``feeding.max_hours`` (or the nominal length) is reached
   first, it stops with status ``timeout``.

Own life: it starts from ``agent.start`` (``common_start.pt``): mind, actor, critic,
normalization, both optimizers and the curiosity running mean, all identical. Only
the temperament and the seed change, and the reward scaler starts afresh: it
measured the returns of another mix of drives. (With ``agent.start`` null it starts
from part 1 with a new actor and critic, as before 21 Nov 2026.) It lives in its
history with its temperament: ``life.n_envs`` bodies in parallel, all in the same
history, each restarting with a new seed and full energy every
``life.episode_steps`` steps (staggered).

One iteration:

1. **Collect** ``life.rollout`` steps. Step by step: ``h_t = GRU(h_{t-1}, [o_t, a_{t-1}])``,
   ``a_t ~ pi(h_t)``, the body acts.
2. **Drives.** The mind's prediction of every ``o_{t+1}`` (head on the stored ``h_t``
   and ``a_t``, before learning) gives curiosity; hunger and caution come from the
   observation and the intended action. The rewards are scaled by the running return
   std.
3. **Predictor.** The self-model learns from this experience, with the prediction
   loss only (TBPTT of ``life.seq_len``, starting from the rollout's initial hidden
   state).
4. **PPO** with GAE on the stored ``h_t``. These are the same hidden states the
   policy saw when it acted, so the probability ratio is consistent. The actor and
   critic read them detached. KL and clip fraction are logged to watch the small lag
   between the stored ``h_t`` and the updated GRU.

Episode restarts happen only between iterations (``episode_steps`` is a multiple of
``rollout``). Every rollout is therefore bootstrapped with ``V(h_T)`` and has no
terminal step inside.

Protected childhood (``life.protected_infancy``): the cost of living (basal and
effort, ``BatchedUmbralEnv.metabolism``) starts at ``start_factor`` and rises linearly
to 1 over the first ``fraction`` of the life, so the agent learns to eat while
hunger still presses little. The same for every agent and history; logged as
``metabolism`` and ``basal_effective``. Caution keeps the nominal effort cost: it is
the mind's own dislike of effort, and the mind cannot read the body's metabolism.

Drive contributions: every iteration logs, per drive, the mean, mean absolute value
and std of its *weighted* contribution to the reward (``contrib_<drive>_mean``,
``_abs``, ``_std``) and its share of the total, ``share_<drive>`` = its mean absolute
contribution over the sum of the three; and the hunger gate (``gate_mean``,
``gate_below_0.5``). The ethics counters are also kept per phase:
protected childhood (iterations up to ``fraction * iterations``) and adulthood,
with the lowest energy of any body and the 10th percentile over all bodies and steps (``energy_min``, ``energy_p10``; also logged every iteration): the
mean of the 32 bodies can look healthy while a few of them are exhausted.

Ethics: the experimenter (not the mind) watches the energy of every body. If, after
a warm-up, more than ``ethics.max_low_fraction`` of the life is spent below
``ethics.low_energy``, the life stops with status ``flagged`` and leaves a summary.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from umbral.checkpoint import Checkpoint, load_checkpoint, make_checkpoint, save_checkpoint
from umbral.config import Config
from umbral.mind.agent import Agent, GaussianNoise, build_agent
from umbral.mind.drives import Drives, RewardScaler, Temperament, temperament_from_config
from umbral.mind.model import CHANNELS, build_model, channel_sq_error, n_parameters
from umbral.mind.training import EVAL_SEED_BASE, POLICY_SEED_OFFSET, TRAIN_SEED_LIMIT, peak_memory_mb
from umbral.recorder import Recorder
from umbral.world.batched import BatchedUmbralEnv

COMPONENTS = ("hunger", "curiosity", "caution")
HUNGER_ONLY = Temperament("hambre", 0.0, 0.0)  # the drive of the common feeding phase (not a temperament)
FEEDING_SEED_BASE = 3 * 2**31  # training seeds of the feeding phase: [3*2**31, 4*2**31), disjoint from all others
FEEDING_EVAL_SEED_BASE = EVAL_SEED_BASE + 4_000_000  # its termination criterion (battery: +3_000_000)


def feeding_histories(config: Config, n: int) -> list[str]:
    """``n`` bodies spread evenly over all the histories (in blocks, in history order)."""
    hs = sorted((str(h) for h in config.world.histories), key=int)
    if n % len(hs):
        raise ValueError(f"{n} bodies cannot be spread evenly over {len(hs)} histories")
    return [h for h in hs for _ in range(n // len(hs))]


@torch.no_grad()
def feeding_eval(agent: Agent, config: Config) -> dict[str, Any]:
    """The termination criterion of the common feeding phase.

    ``feeding.eval_envs_per_history`` bodies per history, full cost (metabolism 1),
    full initial energy, ``feeding.eval_steps`` steps, the stochastic policy, fixed
    seeds (``FEEDING_EVAL_SEED_BASE + i``, noise ``+ POLICY_SEED_OFFSET``), no learning.
    Pure with respect to the agent and the global RNGs. A history passes if its
    fraction of steps below 0.1 < ``max_below_01``, below 0.3 < ``max_below_03`` and
    food per 1000 steps >= ``min_food_per_1000``; ``passed`` if every history does.
    """
    fc = config.feeding
    k = int(fc.eval_envs_per_history)
    hist = feeding_histories(config, k * len(config.world.histories))
    n, T = len(hist), int(fc.eval_steps)
    seeds = [FEEDING_EVAL_SEED_BASE + i for i in range(n)]
    env = BatchedUmbralEnv(config, n, hist)
    env.reset(seed=seeds)
    env.metabolism[...] = 1.0
    noise = GaussianNoise(n, [s + POLICY_SEED_OFFSET for s in seeds])
    world = agent.world
    h = world.initial_state(n)
    prev = np.zeros((n, 6))
    o = env.observation()["flat"]
    low, b03, food, esum = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for _ in range(T):
        h = world.step_hidden(torch.as_tensor(o, dtype=torch.float32), torch.as_tensor(prev, dtype=torch.float32), h)
        a, *_ = agent.act(h[0], noise.sample())
        obs, _, _, _, info = env.step(a)
        o, prev = obs["flat"], info["action_intended"]
        e = info["energy"]
        low += e < float(config.ethics.low_energy)
        b03 += e < 0.3
        food += info["food_eaten"]
        esum += e
    out: dict[str, Any] = {"histories": {}, "steps": T, "envs_per_history": k}
    for name in dict.fromkeys(hist):
        rows = np.array([x == name for x in hist])
        steps = int(rows.sum()) * T
        v = {"fraction_below_0.1": float(low[rows].sum() / steps), "fraction_below_0.3": float(b03[rows].sum() / steps),
             "food_per_1000_steps": float(food[rows].sum() / steps * 1000), "mean_energy": float(esum[rows].sum() / steps)}
        v["passed"] = bool(v["fraction_below_0.1"] < float(fc.max_below_01)
                           and v["fraction_below_0.3"] < float(fc.max_below_03)
                           and v["food_per_1000_steps"] >= float(fc.min_food_per_1000))
        out["histories"][name] = v
    out["passed"] = all(v["passed"] for v in out["histories"].values())
    return out


def gae(rewards: np.ndarray, values: np.ndarray, last_value: np.ndarray, gamma: float, lam: float
        ) -> tuple[np.ndarray, np.ndarray]:
    """Advantages and returns (N, T); no terminal steps inside the rollout."""
    N, T = rewards.shape
    adv = np.zeros((N, T))
    last = np.zeros(N)
    for t in reversed(range(T)):
        nv = last_value if t == T - 1 else values[:, t + 1]
        delta = rewards[:, t] + gamma * nv - values[:, t]
        last = delta + gamma * lam * last
        adv[:, t] = last
    return adv, adv + values


def ppo_loss(agent: Agent, h: torch.Tensor, u: torch.Tensor, logp_old: torch.Tensor, adv: torch.Tensor,
             ret: torch.Tensor, clip: float, vf_coef: float, ent_coef: float) -> tuple[torch.Tensor, dict]:
    mu, v = agent.policy(h)
    logp = agent.policy.log_prob(mu, u)
    log_ratio = logp - logp_old
    ratio = log_ratio.exp()
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    pg = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
    vl = 0.5 * ((v - ret) ** 2).mean()
    ent = agent.policy.entropy()
    loss = pg + vf_coef * vl - ent_coef * ent
    with torch.no_grad():
        stats = {"approx_kl": float(((ratio - 1) - log_ratio).mean()),
                 "clip_fraction": float(((ratio - 1).abs() > clip).float().mean()),
                 "policy_loss": float(pg), "value_loss": float(vl), "entropy": float(ent)}
    return loss, stats


def _hist_quantile(hist: list[int] | None, q: float) -> float | None:
    """Quantile of the energies from their 0.01-wide histogram (lower edge of the bin)."""
    if not hist or sum(hist) == 0:
        return None
    c = np.cumsum(hist)
    return float(np.searchsorted(c, q * c[-1]) / 100)


class Life:
    def __init__(self, config: Config, recorder: Recorder, np_rng: np.random.Generator,
                 resume: Checkpoint | None, feeding: bool = False):
        self.config, self.rec = config, recorder
        self.feeding = bool(feeding)
        lc, pc = config.life, config.ppo
        self.N, self.T = int(lc.n_envs), int(lc.rollout)
        self.iterations = int(lc.iterations)
        self.log_every, self.ckpt_every = int(lc.log_every), int(lc.checkpoint_every)
        if int(lc.episode_steps) % self.T:
            raise ValueError("life.episode_steps must be a multiple of life.rollout")
        if self.T % int(lc.seq_len):
            raise ValueError("life.rollout must be a multiple of life.seq_len")
        if self.ckpt_every % self.log_every:
            raise ValueError("life.checkpoint_every must be a multiple of life.log_every")
        self.mid = self.iterations // 2
        self.infancy_end = int(round(float(lc.protected_infancy.fraction) * self.iterations))
        self.stop_at = None if lc.stop_at is None else int(lc.stop_at)
        self.episode_rollouts = int(lc.episode_steps) // self.T
        if self.feeding:
            self.temperament = HUNGER_ONLY
            self.history = "all"
            self.histories = feeding_histories(config, self.N)
            fc = config.feeding
            self.eval_from = self.infancy_end if fc.eval_from is None else int(fc.eval_from)
            self.eval_every = int(fc.eval_every)
            if self.eval_every % self.log_every or self.eval_from % self.log_every:
                raise ValueError("feeding.eval_every and eval_from must be multiples of life.log_every")
            self.max_seconds = float(fc.max_hours) * 3600
        else:
            self.temperament = temperament_from_config(config, str(config.agent.temperament))
            self.history = str(config.agent.history)
            self.histories = [self.history] * self.N
        self.seed_base = FEEDING_SEED_BASE if self.feeding else 0
        self.env = BatchedUmbralEnv(config, self.N, self.histories)
        layout = self.env.layout

        world = build_model(layout, config.mind, efference_copy=True)
        self.agent = build_agent(world, config.agent)
        self.opts = {"world": torch.optim.Adam(world.parameters(), lr=float(lc.world_lr)),
                     "policy": torch.optim.Adam(self.agent.policy.parameters(), lr=float(pc.lr))}
        self.drives = Drives(layout, config.drives, self.temperament, self.env.params.effort_cost)
        self.scaler = RewardScaler(self.N, float(pc.gamma))
        self.noise = GaussianNoise(self.N, [0] * self.N)

        if resume is None:
            self.np_rng = np_rng
            start = None if self.feeding or not config.agent.start else load_checkpoint(Path(str(config.agent.start)))
            if start is not None:  # the end of the common childhood: all but temperament, seed and reward scaler
                self.agent.load_state_dict(start.models["agent"])
                for k in ("world", "policy"):
                    self.opts[k].load_state_dict(start.optimizers[k])
                self.drives.set_state(start.extra["drives"])
                recorder.update_meta(start=str(config.agent.start), start_step=start.step)
            else:
                infancy = load_checkpoint(Path(str(config.agent.infancy)))
                world.load_state_dict(infancy.models["efference"])
                self.opts["world"].load_state_dict(infancy.optimizers["efference"])
                recorder.update_meta(infancy=str(config.agent.infancy), infancy_step=infancy.step)
            for g in self.opts["world"].param_groups:
                g["lr"] = float(lc.world_lr)
            for g in self.opts["policy"].param_groups:
                g["lr"] = float(pc.lr)
            self.noise = GaussianNoise(self.N, [self._new_seed() for _ in range(self.N)])
            self.env.reset(seed=[self._new_seed() for _ in range(self.N)])
            self.h = world.initial_state(self.N)
            self.prev_action = np.zeros((self.N, 6))
            E = self.episode_rollouts
            self.life_left = np.array([E - (i * E) // self.N for i in range(self.N)])
            self.start, self.elapsed_before = 0, 0.0
            self.ethics = {"steps": 0, "low": 0, "below_03": 0, "food": 0, "energy_sum": 0.0}
            self.energy_curve: list[float] = []
            self.feeding_evals: list[dict[str, Any]] = []
            self.log_std_init = self.agent.policy.log_std.detach().tolist()
        else:
            resume.load_into({"agent": self.agent}, self.opts)
            self.noise.set_state(resume.policy)
            resume.restore_env(self.env)
            self.h = resume.mind_state["h"].clone()
            self.prev_action = resume.mind_state["prev_action"].numpy().astype(float).copy()
            ex = resume.extra
            self.life_left = np.array(ex["life_left"], dtype=int)
            self.drives.set_state(ex["drives"])
            self.scaler.set_state(ex["scaler"])
            self.ethics = dict(ex["ethics"])
            self.energy_curve = list(ex["energy_curve"])
            self.feeding_evals = list(ex.get("feeding_evals", []))
            self.log_std_init = list(ex.get("log_std_init", []))
            self.start, self.elapsed_before = int(resume.step), float(ex["elapsed_seconds"])
            self.np_rng = resume.activate()
        self.times = {"world": 0.0, "mind_acting": 0.0, "predictor": 0.0, "ppo": 0.0, "io": 0.0}
        self._reset_acc()
        recorder.update_meta(
            temperament=self.temperament.name, history=self.history, n_envs=self.N, rollout=self.T,
            feeding_phase=self.feeding, histories_of_bodies=sorted(set(self.histories), key=int),
            n_parameters={"world": n_parameters(world), "policy": n_parameters(self.agent.policy)})

    def _new_seed(self) -> int:
        return self.seed_base + int(self.np_rng.integers(0, TRAIN_SEED_LIMIT))

    def _reset_acc(self) -> None:
        self.acc: dict[str, float] = {}
        self.acc_sq = {c: [0.0, 0] for c in CHANNELS}
        self.acc_n = 0

    def _add(self, k: str, v: float) -> None:
        self.acc[k] = self.acc.get(k, 0.0) + float(v)

    # --- one iteration ------------------------------------------------------------

    def _collect(self) -> dict[str, np.ndarray]:
        rows = np.flatnonzero(self.life_left == 0)
        if len(rows):
            self.env.reset(seed=[self._new_seed() for _ in rows], envs=rows)
            self.h[:, rows] = 0.0
            self.prev_action[rows] = 0.0
            self.scaler.reset_rows(rows)
            self.life_left[rows] = self.episode_rollouts
        N, T, D, H = self.N, self.T, self.env.observation_dim, self.agent.world.hidden_size
        r = {"obs": np.empty((N, T + 1, D)), "act": np.empty((N, T, 6)), "prev": np.empty((N, T, 6)),
             "u": np.empty((N, T, 6), np.float32), "logp": np.empty((N, T), np.float32),
             "value": np.empty((N, T)), "hid": torch.empty(N, T, H), "h0": self.h.clone(),
             "energy": np.empty((N, T)), "food": np.zeros(N), "effort": np.zeros(N)}
        world = self.agent.world
        o = self.env.observation()["flat"]
        r["obs"][:, 0] = o
        tw = tm = 0.0
        for t in range(T):
            ts = time.perf_counter()
            with torch.no_grad():
                self.h = world.step_hidden(torch.as_tensor(o, dtype=torch.float32),
                                           torch.as_tensor(self.prev_action, dtype=torch.float32), self.h)
            a, u, lp, v = self.agent.act(self.h[0], self.noise.sample())
            tm += time.perf_counter() - ts
            ts = time.perf_counter()
            obs, _, _, _, info = self.env.step(a)
            tw += time.perf_counter() - ts
            o = obs["flat"]
            intended = info["action_intended"]
            r["hid"][:, t] = self.h[0]
            r["prev"][:, t], r["act"][:, t] = self.prev_action, intended
            r["u"][:, t], r["logp"][:, t], r["value"][:, t] = u, lp, v
            r["obs"][:, t + 1] = o
            r["energy"][:, t] = info["energy"]
            r["food"] += info["food_eaten"]
            r["effort"] += (info["action_executed"] ** 2).mean(-1)
            self.prev_action = intended
        ts = time.perf_counter()
        with torch.no_grad():  # bootstrap: h_T is recomputed identically at the next iteration's first step
            h_T = world.step_hidden(torch.as_tensor(o, dtype=torch.float32),
                                    torch.as_tensor(self.prev_action, dtype=torch.float32), self.h)
            r["last_value"] = self.agent.policy(h_T[0])[1].numpy().astype(float)
        tm += time.perf_counter() - ts
        self.life_left -= 1
        self.times["world"] += tw
        self.times["mind_acting"] += tm
        return r

    def metabolism(self, it: int) -> float:
        """Protected childhood: factor on the cost of living, rising linearly to 1."""
        pi = self.config.life.protected_infancy
        span = float(pi.fraction) * self.iterations
        if span <= 0:
            return 1.0
        start = float(pi.start_factor)
        return start + (1.0 - start) * min(1.0, (it - 1) / span)

    def step(self, it: int) -> None:
        self.env.metabolism[...] = self.metabolism(it)
        r = self._collect()
        lc, pc = self.config.life, self.config.ppo
        world, layout = self.agent.world, self.env.layout
        ts = time.perf_counter()
        obs = torch.as_tensor(r["obs"], dtype=torch.float32)
        cur, nxt = obs[:, :-1], obs[:, 1:]
        act = torch.as_tensor(r["act"], dtype=torch.float32)
        prev = torch.as_tensor(r["prev"], dtype=torch.float32)
        with torch.no_grad():  # the prediction the mind made when acting (before learning from it)
            pred = world.predict_next(cur, world.from_hidden(cur, r["hid"], act))
        rewards, comps = self.drives(r["obs"][:, :-1], r["obs"][:, 1:], pred.double().numpy(), r["act"])
        scaled = self.scaler(rewards)
        adv, ret = gae(scaled, r["value"], r["last_value"], float(pc.gamma), float(pc.lam))
        sq = channel_sq_error(layout, pred, nxt)

        # Predictor: prediction loss only.
        L = int(lc.seq_len)
        h = r["h0"]
        for s in range(0, self.T, L):
            out = world(cur[:, s:s + L], prev[:, s:s + L], act[:, s:s + L], h)
            loss = world.loss(cur[:, s:s + L], nxt[:, s:s + L], out)["total"]
            self.opts["world"].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(world.parameters(), float(lc.grad_clip))
            self.opts["world"].step()
            h = out.h_last.detach()
            self._add("world_loss", float(loss.detach()) * L / self.T)
        t1 = time.perf_counter()
        self.times["predictor"] += t1 - ts

        # PPO on the stored hidden states (actor and critic only).
        n = self.N * self.T
        flat = {"h": r["hid"].reshape(n, -1), "u": torch.as_tensor(r["u"]).reshape(n, 6),
                "logp": torch.as_tensor(r["logp"]).reshape(n),
                "adv": torch.as_tensor(adv, dtype=torch.float32).reshape(n),
                "ret": torch.as_tensor(ret, dtype=torch.float32).reshape(n)}
        mb = n // int(pc.minibatches)
        ppo_stats: dict[str, float] = {}
        count = 0
        for _ in range(int(pc.epochs)):
            perm = torch.as_tensor(self.np_rng.permutation(n))
            for k in range(int(pc.minibatches)):
                idx = perm[k * mb:(k + 1) * mb]
                loss, st = ppo_loss(self.agent, flat["h"][idx], flat["u"][idx], flat["logp"][idx],
                                    flat["adv"][idx], flat["ret"][idx], float(pc.clip), float(pc.vf_coef),
                                    float(pc.ent_coef))
                self.opts["policy"].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.policy.parameters(), float(pc.grad_clip))
                self.opts["policy"].step()
                for key, v in st.items():
                    ppo_stats[key] = ppo_stats.get(key, 0.0) + v
                count += 1
        t2 = time.perf_counter()
        self.times["ppo"] += t2 - t1

        # Weighted drive contributions, every iteration.
        abs_means = {k: float(np.abs(comps["weighted"][k]).mean()) for k in COMPONENTS}
        abs_total = sum(abs_means.values()) or 1.0
        for k in COMPONENTS:
            w = comps["weighted"][k]
            self.rec.log_metric(it, f"contrib_{k}_mean", float(w.mean()))
            self.rec.log_metric(it, f"contrib_{k}_abs", abs_means[k])
            self.rec.log_metric(it, f"contrib_{k}_std", float(w.std()))
            self.rec.log_metric(it, f"share_{k}", abs_means[k] / abs_total)
        # The 32 bodies one by one, not only their mean: lowest energy and 10th percentile this iteration.
        self.rec.log_metric(it, "energy_min", float(r["energy"].min()))
        self.rec.log_metric(it, "energy_p10", float(np.percentile(r["energy"], 10)))
        self.rec.log_metric(it, "gate_mean", float(comps["gate"].mean()))
        self.rec.log_metric(it, "gate_below_0.5", float((comps["gate"] < 0.5).mean()))

        # Accumulate metrics and the ethics counters (experimenter side: true energy).
        e = r["energy"]
        phase = "infancy" if it <= self.infancy_end else "adult"
        for key, v in (("steps", e.size), ("low", int((e < float(self.config.ethics.low_energy)).sum())),
                       ("below_03", int((e < 0.3).sum())), ("food", int(r["food"].sum())),
                       ("energy_sum", float(e.sum()))):
            self.ethics[f"{key}_{phase}"] = self.ethics.get(f"{key}_{phase}", 0) + v
        hist = np.bincount(np.clip((e * 100).astype(int), 0, 100).ravel(), minlength=101)
        self.ethics[f"hist_{phase}"] = (np.asarray(self.ethics.get(f"hist_{phase}", np.zeros(101, int))) + hist).tolist()
        self.ethics[f"min_{phase}"] = min(float(self.ethics.get(f"min_{phase}", 1.0)), float(e.min()))
        self.ethics["steps"] += e.size
        self.ethics["low"] += int((e < float(self.config.ethics.low_energy)).sum())
        self.ethics["below_03"] += int((e < 0.3).sum())
        self.ethics["food"] += int(r["food"].sum())
        self.ethics["energy_sum"] += float(e.sum())
        if it > self.iterations * float(self.config.ethics.warmup_fraction):
            self.ethics["steps_checked"] = self.ethics.get("steps_checked", 0) + e.size
            self.ethics["low_checked"] = self.ethics.get("low_checked", 0) + int(
                (e < float(self.config.ethics.low_energy)).sum())
        for c in CHANNELS:
            self.acc_sq[c][0] += sq[c][0]
            self.acc_sq[c][1] += sq[c][1]
        for k in COMPONENTS:
            self._add(f"reward_{k}", comps[k].mean())
        self._add("curiosity_raw", comps["curiosity_raw"].mean())
        self._add("reward_total", rewards.mean())
        self._add("energy", e.mean())
        self._add("energy_below_0.3", (e < 0.3).mean())
        self._add("energy_below_0.1", (e < float(self.config.ethics.low_energy)).mean())
        self._add("food_per_1000_steps", r["food"].sum() / e.size * 1000)
        self._add("effort", r["effort"].sum() / e.size)
        self._add("metabolism", float(self.env.metabolism))
        self._add("basal_effective", float(self.env.metabolism) * self.env.params.basal)
        self._add("action_std", float(self.agent.policy.log_std.detach().exp().mean()))
        for key, v in ppo_stats.items():
            self._add(f"ppo_{key}", v / count)
        self.acc_n += 1
        self.times["io"] += time.perf_counter() - t2

    def _log(self, it: int) -> None:
        for k, v in self.acc.items():
            self.rec.log_metric(it, k, v / self.acc_n)
        for c in CHANNELS:
            self.rec.log_metric(it, f"mse_{c}", self.acc_sq[c][0] / self.acc_sq[c][1])
        self.energy_curve.append(round(self.acc["energy"] / self.acc_n, 5))
        self._reset_acc()

    # --- checkpoints and loop -----------------------------------------------------

    def checkpoint(self, it: int, name: str, elapsed: float) -> Path:
        ckpt = make_checkpoint(
            it, {"agent": self.agent}, self.opts, self.np_rng, self.config,
            extra={"life_left": self.life_left.tolist(), "drives": self.drives.get_state(),
                   "scaler": {**self.scaler.get_state(), "ret": self.scaler.ret.tolist()},
                   "ethics": dict(self.ethics), "energy_curve": list(self.energy_curve),
                   "elapsed_seconds": elapsed, "temperament": self.temperament.name, "history": self.history,
                   "feeding_evals": list(self.feeding_evals), "log_std_init": list(self.log_std_init)},
            env=self.env, policy=self.noise,
            mind_state={"h": self.h, "prev_action": torch.as_tensor(self.prev_action)})
        return save_checkpoint(self.rec.run_dir / "checkpoints" / f"{name}.pt", ckpt)

    def ethics_summary(self) -> dict[str, Any]:
        e = self.ethics
        steps = max(e["steps"], 1)
        return {"steps": e["steps"], "fraction_below_0.1": e["low"] / steps,
                "fraction_below_0.3": e["below_03"] / steps,
                "fraction_below_0.1_after_warmup": e.get("low_checked", 0) / max(e.get("steps_checked", 0), 1),
                "food_eaten": e["food"], "food_per_1000_steps": e["food"] / steps * 1000,
                "mean_energy": e["energy_sum"] / steps,
                "phases": {ph: {"steps": e.get(f"steps_{ph}", 0),
                                "fraction_below_0.1": e.get(f"low_{ph}", 0) / max(e.get(f"steps_{ph}", 0), 1),
                                "fraction_below_0.3": e.get(f"below_03_{ph}", 0) / max(e.get(f"steps_{ph}", 0), 1),
                                "mean_energy": e.get(f"energy_sum_{ph}", 0.0) / max(e.get(f"steps_{ph}", 0), 1),
                                "food_per_1000_steps": e.get(f"food_{ph}", 0) / max(e.get(f"steps_{ph}", 0), 1) * 1000,
                                "energy_min": e.get(f"min_{ph}"),
                                "energy_p10": _hist_quantile(e.get(f"hist_{ph}"), 0.10)}
                           for ph in ("infancy", "adult")},
                "infancy_end_iteration": self.infancy_end,
                "energy_curve": list(self.energy_curve),
                "energy_curve_every_iterations": self.log_every}

    def run(self) -> dict[str, Any]:
        wall0 = time.perf_counter()
        done_here = 0
        status = "completed"
        end = self.iterations if self.stop_at is None else min(self.stop_at, self.iterations)
        for it in range(self.start + 1, end + 1):
            self.step(it)
            done_here += 1
            elapsed = self.elapsed_before + time.perf_counter() - wall0
            if it % self.log_every == 0 or it == end:
                self._log(it)
                per = (time.perf_counter() - wall0) / done_here
                self.rec.log(f"iteration {it}/{self.iterations}  {per:.2f} s/it  "
                             f"{self.N * self.T / per:.0f} steps/s  energy {self.energy_curve[-1]:.3f}  "
                             f"ETA {per * (self.iterations - it) / 60:.1f} min")
                chk = self.ethics
                if chk.get("steps_checked", 0) and chk["low_checked"] / chk["steps_checked"] > float(
                        self.config.ethics.max_low_fraction):
                    self.checkpoint(it, f"iter_{it:07d}", elapsed)
                    status = "flagged"
                    break
                if self.feeding and it >= self.eval_from and (it - self.eval_from) % self.eval_every == 0:
                    ev = feeding_eval(self.agent, self.config)
                    ev["iteration"] = it
                    self.feeding_evals.append(ev)
                    for name, v in ev["histories"].items():
                        for key in ("fraction_below_0.1", "fraction_below_0.3", "food_per_1000_steps", "mean_energy"):
                            self.rec.log_metric(it, f"criterion_h{name}_{key}", v[key])
                    self.rec.log(f"criterion at {it}: " + "  ".join(
                        f"h{name} <0.1 {v['fraction_below_0.1']:.4f} <0.3 {v['fraction_below_0.3']:.4f} "
                        f"food {v['food_per_1000_steps']:.2f} {'ok' if v['passed'] else 'no'}"
                        for name, v in ev["histories"].items()))
                    if ev["passed"]:
                        self.checkpoint(it, "common_start", elapsed)
                        status = "completed"
                        break
                if self.feeding and elapsed >= self.max_seconds:
                    self.checkpoint(it, f"iter_{it:07d}", elapsed)
                    status = "timeout"
                    break
            if it == self.mid and not self.feeding:
                self.checkpoint(it, "life_short", elapsed)
            if it == self.iterations and not self.feeding:
                self.checkpoint(it, "life_long", elapsed)
            elif it % self.ckpt_every == 0 or it == end:
                self.checkpoint(it, f"iter_{it:07d}", elapsed)
        else:
            if end < self.iterations:
                status = "paused"
            elif self.feeding:
                status = "timeout"  # the whole nominal length without meeting the criterion
        wall = time.perf_counter() - wall0
        total = sum(self.times.values()) or 1.0
        summary = {
            "life_status": status,
            "iterations_this_run": done_here,
            "wall_seconds": round(wall, 1),
            "life_seconds": round(self.elapsed_before + wall, 1),
            "env_steps_per_second": round(done_here * self.N * self.T / max(wall, 1e-9), 1),
            "time_fractions": {k: round(v / total, 4) for k, v in self.times.items()},
            "peak_memory_mb": None if peak_memory_mb() is None else round(peak_memory_mb(), 1),
            "ethics": self.ethics_summary(),
        }
        if self.feeding:
            summary["feeding"] = {"evals": list(self.feeding_evals), "log_std_init": list(self.log_std_init),
                                  "log_std_final": self.agent.policy.log_std.detach().tolist()}
        self.rec.update_meta(**summary)
        return summary
