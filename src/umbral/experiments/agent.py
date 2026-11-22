"""Phase 2b experiments: ``live_agent``, ``behavior_battery`` and ``agent_report``.

- ``live_agent``: one agent's own life (``umbral.mind.life``) from the common
  childhood. Leaves ``checkpoints/life_short.pt`` (half) and ``life_long.pt`` (end),
  and, with ``life.run_report``, ``behavior.json`` and ``prediction_check.json`` for
  the final agent. If the ethics rule stops it: ``flagged_summary.json`` and
  ``life_status: flagged`` in meta.json. The battery is also run on ``life_short.pt``
  (``behavior_short.json``), to see whether the character settles.
- ``behavior_battery``: the battery (``umbral.metrics.behavior``) for the agent in
  ``behavior.checkpoint``.
- ``agent_report``: compares the agents in ``report.agents`` (run folders): behaviour
  with CIs and effect sizes at the short and the long life, ethics and energy per
  phase (protected childhood / adulthood), each drive's share of the reward per phase,
  prediction check; ``report.json`` and figures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from umbral.checkpoint import Checkpoint, load_checkpoint
from umbral.config import Config
from umbral.experiments.registry import register
from umbral.metrics.behavior import MEASURES, compare, run_battery
from umbral.mind.agent import Agent, build_agent
from umbral.mind.evaluation import AgentActor, bootstrap_ratio_ci, evaluate
from umbral.mind.life import COMPONENTS, Life
from umbral.mind.model import CHANNELS, build_model
from umbral.mind.training import EVAL_SEED_BASE, POLICY_SEED_OFFSET
from umbral.recorder import Recorder


def load_agent(ckpt: Checkpoint) -> tuple[Agent, Config]:
    from umbral.body.sensors import ObsLayout

    cfg = Config(ckpt.config)
    world = build_model(ObsLayout(int(cfg.body.vision_rays)), cfg.mind, efference_copy=True)
    agent = build_agent(world, cfg.agent)
    agent.load_state_dict(ckpt.models["agent"])
    agent.eval()
    return agent, cfg


def run_chain(run_dir: Path) -> list[Path]:
    """The run folders of one life, first to last, following ``run.resume_from``."""
    chain = [Path(run_dir)]
    while True:
        cfg_path = chain[0] / "config.yaml"
        if not cfg_path.is_file():
            break
        prev = (yaml.safe_load(cfg_path.read_text(encoding="utf-8")).get("run") or {}).get("resume_from")
        if not prev:
            break
        d = Path(str(prev)).parent.parent
        if d in chain or not d.is_dir():
            break
        chain.insert(0, d)
    return chain


def find_checkpoint(run_dir: Path, name: str) -> Path | None:
    """The life's checkpoint ``name``, searched from the last run folder backwards."""
    for d in reversed(run_chain(run_dir)):
        p = d / "checkpoints" / f"{name}.pt"
        if p.is_file():
            return p
    return None


def drive_shares(run_dir: Path, infancy_end: int) -> dict[str, Any]:
    """Per phase, each drive's share of the reward: mean |weighted contribution| over their sum.

    Also the mean signed contribution and the mean within-iteration std. Phases:
    protected childhood (iterations <= ``infancy_end``) and adulthood.
    """
    stats = {s: {ph: {k: 0.0 for k in COMPONENTS} for ph in ("infancy", "adult")} for s in ("abs", "mean", "std")}
    iters: dict[str, set] = {"infancy": set(), "adult": set()}
    for d in run_chain(run_dir):
        mp = d / "metrics.jsonl"
        if not mp.is_file():
            continue
        for line in mp.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if not row["name"].startswith("contrib_"):
                continue
            k, stat = row["name"][len("contrib_"):].rsplit("_", 1)
            ph = "infancy" if row["step"] <= infancy_end else "adult"
            iters[ph].add(row["step"])
            stats[stat][ph][k] += row["value"]
    out: dict[str, Any] = {}
    for ph in ("infancy", "adult"):
        n = len(iters[ph])
        if not n:
            out[ph] = None
            continue
        tot = sum(stats["abs"][ph].values()) or 1.0
        out[ph] = {"iterations": n, "share_abs": {k: stats["abs"][ph][k] / tot for k in COMPONENTS},
                   **{f"mean_{s}": {k: stats[s][ph][k] / n for k in COMPONENTS} for s in ("abs", "mean", "std")}}
    return out


def prediction_check(world, baseline_path: str | Path, config: Config, agent: Agent | None = None) -> dict[str, Any]:
    """The agent's self-model vs the childhood model, Phase 2a protocol (OU policy, eval seeds).

    Both are evaluated on the same environments, in the *current* world physics, so the
    baseline is always measured in the same world as the agent. Ratio agent / childhood
    per channel with a 95 % bootstrap CI over seeds (stratified by history).
    With ``agent``, the data come from that agent's own stochastic policy instead of the
    OU policy (same seeds; ``prediction_own``): both minds predict the agent's own life.
    """
    base = load_checkpoint(Path(str(baseline_path)))
    bcfg = Config(base.config)
    w0 = build_model(world.layout, bcfg.mind, efference_copy=True)
    w0.load_state_dict(base.models["efference"])
    w0.eval()
    ec = config.eval
    hids = sorted(config.world.histories)
    histories = [h for h in hids for _ in range(int(ec.envs_per_history))]
    n = len(histories)
    world.eval()
    pseeds = [EVAL_SEED_BASE + POLICY_SEED_OFFSET + i for i in range(n)]
    actor = None if agent is None else AgentActor(agent, n, pseeds)
    sums = evaluate({"agent": world, "childhood": w0}, config, histories, [EVAL_SEED_BASE + i for i in range(n)],
                    pseeds, int(ec.steps), int(ec.chunk), actor=actor)
    rows = sums["rows"]
    out: dict[str, Any] = {"baseline": str(baseline_path), "envs": n, "steps": int(ec.steps),
                           "behaviour": "ou" if agent is None else "own", "channels": {}}
    for c in CHANNELS:
        a, b = rows["mse"]["agent"][c], rows["mse"]["childhood"][c]
        ma, mb = a[:, 0].sum() / a[:, 1].sum(), b[:, 0].sum() / b[:, 1].sum()
        lo, hi = bootstrap_ratio_ci(a, b, histories, int(ec.n_boot), int(ec.bootstrap_seed))
        out["channels"][c] = {"agent": float(ma), "childhood": float(mb), "ratio": float(ma / mb),
                              "ci95": [lo, hi]}
        out["channels"][c]["by_history"] = {
            hid: float(sums["mse"]["agent"][hid][c][0] / sums["mse"]["childhood"][hid][c][0]) for hid in hids}
    return out


@register("live_agent")
def run_live_agent(config: Config, recorder: Recorder, np_rng: np.random.Generator,
                   resume: Checkpoint | None) -> None:
    life = Life(config, recorder, np_rng, resume)
    summary = life.run()
    d = recorder.run_dir
    if summary["life_status"] == "flagged":
        (d / "flagged_summary.json").write_text(json.dumps(summary["ethics"], indent=2) + "\n", encoding="utf-8")
        recorder.log(f"FLAGGED by the ethics rule ({summary['flag_reason']}): "
                     f"{summary['ethics']['fraction_below_0.1_after_warmup']:.3f} of the life below "
                     f"{config.ethics.low_energy} after warm-up; longest stretch at energy 0 of one body "
                     f"{summary['ethics']['zero_runs']['longest']} steps (limit {config.ethics.zero_run_limit})")
        return
    recorder.log(f"life {summary['life_status']}: {summary['env_steps_per_second']} steps/s, "
                 f"time fractions {summary['time_fractions']}")
    if summary["life_status"] == "completed" and bool(config.life.run_report):
        agent = life.agent
        agent.eval()
        beh = run_battery(agent, config, str(config.agent.history))
        (d / "behavior.json").write_text(json.dumps(beh, indent=2) + "\n", encoding="utf-8")
        review = {"behavior": battery_review(beh, config)}
        short = find_checkpoint(d, "life_short")
        if short is not None:
            agent_s, _ = load_agent(load_checkpoint(short))
            beh_s = run_battery(agent_s, config, str(config.agent.history))
            beh_s["checkpoint"] = str(short)
            (d / "behavior_short.json").write_text(json.dumps(beh_s, indent=2) + "\n", encoding="utf-8")
            review["behavior_short"] = battery_review(beh_s, config)
        if any(review.values()):  # the individual rule in the battery: stops nothing, marks for review
            recorder.update_meta(battery_review=review)
            recorder.log(f"REVIEW: battery episodes with a stretch at energy 0 >= {config.ethics.zero_run_limit} "
                         f"steps: {review}")
        pc = prediction_check(agent.world, config.agent.infancy, config)
        (d / "prediction_check.json").write_text(json.dumps(pc, indent=2) + "\n", encoding="utf-8")
        for c, v in pc["channels"].items():
            recorder.log(f"prediction {c}: agent/childhood {v['ratio']:.3f} "
                         f"[{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
        own = prediction_check(agent.world, config.agent.infancy, config, agent=agent)  # own behaviour (22 Nov)
        (d / "prediction_own.json").write_text(json.dumps(own, indent=2) + "\n", encoding="utf-8")
        for c, v in own["channels"].items():
            recorder.log(f"prediction on own behaviour {c}: agent/childhood {v['ratio']:.3f} "
                         f"[{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")


@register("common_feeding")
def run_common_feeding(config: Config, recorder: Recorder, np_rng: np.random.Generator,
                       resume: Checkpoint | None) -> None:
    """Common childhood, part 2: learning to eat (see ``umbral.mind.life``).

    Leaves ``checkpoints/common_start.pt`` if the criterion is met, and
    ``feeding_summary.json``: every criterion evaluation per history, and the actor's
    ``log_std`` at its initialization and at the end (how much exploration the phase
    took away), plus the ethics summary.
    """
    from umbral.mind.report import plot_feeding, plot_feeding_curves

    life = Life(config, recorder, np_rng, resume, feeding=True)
    summary = life.run()
    d = recorder.run_dir
    f = summary["feeding"]
    init, final = np.array(f["log_std_init"]), np.array(f["log_std_final"])
    out = {"status": summary["life_status"], "common_start": str(d / "checkpoints" / "common_start.pt")
           if summary["life_status"] == "completed" else None,
           "evals": f["evals"], "cost_factor_final": f["cost_factor_final"],
           "full_cost_from_iteration": f["full_cost_from_iteration"],
           "log_std": {"init": init.tolist(), "final": final.tolist(), "init_mean": float(init.mean()),
                       "final_mean": float(final.mean()), "std_init_mean": float(np.exp(init).mean()),
                       "std_final_mean": float(np.exp(final).mean()),
                       "std_ratio_final_over_init": float(np.exp(final).mean() / np.exp(init).mean())},
           "ethics": {k: v for k, v in summary["ethics"].items() if k != "energy_curve"}}
    (d / "feeding_summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    if summary["life_status"] == "flagged":
        (d / "flagged_summary.json").write_text(json.dumps(summary["ethics"], indent=2) + "\n", encoding="utf-8")
        recorder.log(f"FLAGGED by the ethics rule ({summary['flag_reason']}): "
                     f"{summary['ethics']['fraction_below_0.1_after_warmup']:.3f} of the phase below "
                     f"{config.ethics.low_energy} after warm-up; longest stretch at energy 0 of one body "
                     f"{summary['ethics']['zero_runs']['longest']} steps")
    recorder.log(f"feeding {summary['life_status']}: action std {out['log_std']['std_init_mean']:.3f} -> "
                 f"{out['log_std']['std_final_mean']:.3f} (x{out['log_std']['std_ratio_final_over_init']:.2f})")
    recorder.log(f"cost factor at the end {out['cost_factor_final']:.3f}; full cost from iteration "
                 f"{out['full_cost_from_iteration']}")
    if f["evals"]:
        plot_feeding(f["evals"], config, d / "feeding_criterion.png")
    plot_feeding_curves(run_chain(d), d / "feeding_curves.png")


def battery_review(beh: dict[str, Any], config: Config) -> list[int]:
    """Battery episodes whose longest stretch at energy 0 reaches ``ethics.zero_run_limit``.

    The individual rule in the battery: it stops nothing (the battery does not learn), but
    the episodes are marked and the agent is left for review before going on.
    """
    limit = config.ethics.zero_run_limit
    if limit is None:
        return []
    return [i for i, r in enumerate(beh.get("energy_zero_run", [])) if r >= int(limit)]


def energy_zero_summary(beh: dict[str, list[float]] | None) -> dict[str, Any] | None:
    """How many battery episodes reach energy 0, and how long they stay there."""
    if not beh or "energy_zero" not in beh:
        return None
    z, run = np.array(beh["energy_zero"]), np.array(beh["energy_zero_run"])
    hit = z > 0
    return {"episodes": int(len(z)), "episodes_reaching_zero": int(hit.sum()),
            "mean_fraction_at_zero": float(z.mean()),
            "mean_fraction_at_zero_when_reached": float(z[hit].mean()) if hit.any() else 0.0,
            "longest_run_steps": float(run.max()) if len(run) else 0.0}


@register("prediction_own")
def run_prediction_own(config: Config, recorder: Recorder, np_rng: np.random.Generator,
                       resume: Checkpoint | None) -> None:
    """Prediction error on the agent's own behaviour, besides the Phase 2a protocol (informative).

    For every checkpoint in ``prediction.checkpoints``: the agent's mind and the childhood
    model (``agent.infancy``) predict (a) the OU policy of Phase 2a and (b) the agent's own
    stochastic policy, on the same evaluation seeds and the 4 histories. Ratio agent /
    childhood per channel with 95 % CIs, per history, and the agent's own-vs-OU error.
    Writes ``prediction_own.json``. It changes no criterion.
    """
    out: dict[str, Any] = {"baseline": str(config.agent.infancy), "checkpoints": {}}
    for path in [str(p) for p in config.prediction.checkpoints]:
        agent, _ = load_agent(load_checkpoint(Path(path)))
        res = {"ou": prediction_check(agent.world, config.agent.infancy, config),
               "own": prediction_check(agent.world, config.agent.infancy, config, agent=agent)}
        res["agent_own_over_ou"] = {c: res["own"]["channels"][c]["agent"] / res["ou"]["channels"][c]["agent"]
                                    for c in CHANNELS}
        res["childhood_own_over_ou"] = {c: res["own"]["channels"][c]["childhood"]
                                        / res["ou"]["channels"][c]["childhood"] for c in CHANNELS}
        out["checkpoints"][path] = res
        for b in ("ou", "own"):
            recorder.log(f"{path} [{b}] " + "  ".join(
                f"{c} {v['ratio']:.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]"
                for c, v in res[b]["channels"].items()))
        (recorder.run_dir / "prediction_own.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


@register("start_robustness")
def run_start_robustness(config: Config, recorder: Recorder, np_rng: np.random.Generator,
                         resume: Checkpoint | None) -> None:
    """The feeding criterion again on ``agent.start``, with other seed blocks (informative only).

    One ``feeding_eval`` per ``feeding.robustness_offsets`` entry (seeds
    ``FEEDING_EVAL_SEED_BASE + offset + i``), plus the original block (offset 0) for
    reference. Writes ``robustness.json``; it decides nothing.
    """
    from umbral.mind.life import feeding_eval

    if not config.agent.start:
        raise ValueError("agent.start must point to common_start.pt")
    ckpt = load_checkpoint(Path(str(config.agent.start)))
    agent, _ = load_agent(ckpt)
    out = {"checkpoint": str(config.agent.start), "step": ckpt.step, "evals": []}
    for off in [0, *[int(o) for o in config.feeding.robustness_offsets]]:
        ev = feeding_eval(agent, config, off)
        out["evals"].append(ev)
        for h, v in ev["histories"].items():
            recorder.log(f"offset {off} h{h}: below 0.1 {v['fraction_below_0.1']:.4f}  below 0.3 "
                         f"{v['fraction_below_0.3']:.4f}  food/1000 {v['food_per_1000_steps']:.3f}  "
                         f"{'ok' if v['passed'] else 'FAIL'}")
        recorder.log(f"offset {off}: {'passed' if ev['passed'] else 'not passed'}")
    recorder.update_meta(agent_checkpoint=str(config.agent.start), agent_step=ckpt.step)
    (recorder.run_dir / "robustness.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


def zero_runs(e: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) of every stretch with energy exactly 0 (end exclusive)."""
    z = np.r_[False, e == 0.0, False]
    d = np.diff(z.astype(int))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def analyze_zero_episode(sig: dict[str, np.ndarray], low: float = 0.1) -> dict[str, Any]:
    """What happens in a recorded battery episode around its longest stretch at energy 0."""
    e = sig["energy"].astype(float)
    runs = zero_runs(e)
    T = len(e)
    pos = sig["trunk_pose"][:, :2].astype(float)
    speed = np.r_[0.0, np.sqrt((np.diff(pos, axis=0) ** 2).sum(-1))]
    effort = (sig["action_executed"].astype(float) ** 2).mean(-1)
    intended = np.abs(sig["action_intended"].astype(float)).mean(-1)
    mean_mag = np.abs(sig["action_mean"].astype(float)).mean(-1)
    fp, fa = sig["food_pos"].astype(float), sig["food_active"].astype(bool)
    dist = np.sqrt(((fp - pos[:, None]) ** 2).sum(-1))
    nearest_food = np.where(fa.any(-1), np.where(fa, dist, np.inf).min(-1), np.nan)
    size = float(sig["arena_size"])
    wall = np.minimum(pos, size - pos).min(-1)
    op, orr = sig["obstacle_pos"].astype(float), sig["obstacle_r"].astype(float)
    obst = (np.sqrt(((op[None] - pos[:, None]) ** 2).sum(-1)) - orr[None]).min(-1) if len(orr) else np.full(T, np.nan)
    touch = sig["raw_flat"]  # contact types from the touch rows
    eaten = sig["food_eaten"].astype(float)

    def seg(a: np.ndarray, s: int, t: int) -> float:
        v = a[s:t]
        v = v[np.isfinite(v)]
        return float(v.mean()) if len(v) else float("nan")

    out: dict[str, Any] = {"steps": T, "env_seed": int(sig["env_seed"]), "history": str(sig["history_id"]),
                           "zero_runs": runs, "fraction_at_zero": float((e == 0).mean()),
                           "longest_run": max((t - s for s, t in runs), default=0),
                           "first_below_low": int(np.argmax(e < low)) if (e < low).any() else None,
                           "food_total": float(eaten.sum())}
    if not runs:
        return out
    s, t = max(runs, key=lambda x: x[1] - x[0])
    pre = max(0, s - 500)
    out["longest"] = {
        "start": s, "end": t, "ends_by": "episode end" if t == T else ("food" if eaten[t - 1:t + 1].sum() > 0
                                                                       else "energy rose"),
        "displacement": float(np.linalg.norm(pos[t - 1] - pos[s])),
        "path_length": float(speed[s:t].sum()),
        "box": [float(np.ptp(pos[s:t, 0])), float(np.ptp(pos[s:t, 1]))],
    }
    for name, a in (("speed", speed), ("effort", effort), ("intended_abs", intended), ("policy_mean_abs", mean_mag),
                    ("nearest_food", nearest_food), ("wall_distance", wall), ("obstacle_distance", obst),
                    ("contacts", sig["contacts"].astype(float)), ("food_active", fa.sum(-1).astype(float))):
        out["longest"][name] = {"before_500": seg(a, pre, s), "during": seg(a, s, t),
                                "healthy": seg(np.where(e > 0.5, a, np.nan), 0, T)}
    out["longest"]["food_eaten_during"] = float(eaten[s:t].sum())
    out["longest"]["energy_before"] = [float(x) for x in e[max(0, s - 1000):s:100]]
    return out


@register("battery_replay")
def run_battery_replay(config: Config, recorder: Recorder, np_rng: np.random.Generator,
                       resume: Checkpoint | None) -> None:
    """Battery episodes again, recorded step by step, with an analysis of their energy-0 stretches.

    Same agent (``behavior.checkpoint``), history (``behavior.history``, default the
    agent's) and seeds as the battery: the measures must match the battery's exactly
    (``replay_matches`` in ``replay.json``). For every index in ``replay.episodes``:
    ``signals_ep<i>.npz`` (the viewer's format: ``umbral.world.render.save_gif``),
    ``ep<i>.gif`` around the longest stretch at energy 0, and its analysis.
    """
    from umbral.world.render import save_gif

    ckpt = load_checkpoint(Path(str(config.behavior.checkpoint)))
    agent, acfg = load_agent(ckpt)
    history = str(acfg.agent.history if config.behavior.history is None else config.behavior.history)
    eps = [int(i) for i in config.replay.episodes]
    beh = run_battery(agent, config, history, record=eps)
    recs = beh.pop("recordings")
    ref_path = Path(str(config.behavior.checkpoint)).parent.parent
    out: dict[str, Any] = {"checkpoint": str(config.behavior.checkpoint), "history": history, "episodes": {}}
    for i in eps:
        sig = recs[i]
        np.savez_compressed(recorder.run_dir / f"signals_ep{i}.npz", **sig)
        a = analyze_zero_episode(sig, float(config.ethics.low_energy))
        a["battery"] = {k: v[i] for k, v in beh.items() if isinstance(v, list)}
        out["episodes"][i] = a
        if a.get("longest"):
            s = a["longest"]["start"]
            clip = {k: (v[max(0, s - 300):] if getattr(v, "ndim", 0) and len(v) == a["steps"] else v)
                    for k, v in sig.items()}
            save_gif(clip, recorder.run_dir / f"ep{i}.gif", steps=min(1200, a["steps"] - max(0, s - 300)), stride=6,
                     fps=15, title=f"h{history} ep{i} · from step {max(0, s - 300)} ·")
        recorder.log(f"episode {i}: at zero {a['fraction_at_zero']:.3f}, longest {a['longest_run']} steps, "
                     f"runs {len(a['zero_runs'])}")
    out["measures"] = beh
    (recorder.run_dir / "replay.json").write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")
    recorder.update_meta(agent_checkpoint=str(config.behavior.checkpoint), history=history, episodes=eps)


@register("behavior_battery")
def run_behavior_battery(config: Config, recorder: Recorder, np_rng: np.random.Generator,
                         resume: Checkpoint | None) -> None:
    if not config.behavior.checkpoint:
        raise ValueError("behavior.checkpoint must point to a live_agent checkpoint")
    ckpt = load_checkpoint(Path(str(config.behavior.checkpoint)))
    agent, acfg = load_agent(ckpt)
    history = str(acfg.agent.history if config.behavior.history is None else config.behavior.history)
    beh = run_battery(agent, config, history)
    beh["checkpoint"] = str(config.behavior.checkpoint)
    temperament = (ckpt.extra or {}).get("temperament") or str(acfg.agent.temperament)
    recorder.update_meta(temperament=temperament, history=history,
                         agent_checkpoint=str(config.behavior.checkpoint), agent_step=ckpt.step)
    (recorder.run_dir / "behavior.json").write_text(json.dumps(beh, indent=2) + "\n", encoding="utf-8")


@register("agent_report")
def run_agent_report(config: Config, recorder: Recorder, np_rng: np.random.Generator,
                     resume: Checkpoint | None) -> None:
    from umbral.mind.report import plot_agents

    agents: dict[str, dict[str, Any]] = {}
    for d in config.report.agents:
        d = Path(str(d))
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        name = f"{meta.get('temperament')}-h{meta.get('history')}-{d.name}"
        info: dict[str, Any] = {"dir": str(d), "meta": {k: meta.get(k) for k in (
            "temperament", "history", "life_status", "life_seconds", "env_steps_per_second", "peak_memory_mb")},
            "ethics": {k: v for k, v in (meta.get("ethics") or {}).items() if k != "energy_curve"}}
        for f in ("behavior", "behavior_short", "prediction_check"):
            p = d / f"{f}.json"
            info[f] = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
        info["drive_shares"] = drive_shares(d, int(info["ethics"].get("infancy_end_iteration") or 0))
        agents[name] = info
    report: dict[str, Any] = {"agents": agents}
    for key in ("behavior", "behavior_short"):
        with_beh = {k: v[key] for k, v in agents.items() if v[key]}
        report[key] = compare(with_beh, int(config.behavior.n_boot)) if with_beh else None
    # The character's development: the shared starting point (one battery: before living, the policy
    # does not depend on the temperament) against the end of the short and the long life.
    start = None
    if config.report.start_behavior:
        start = json.loads((Path(str(config.report.start_behavior)) / "behavior.json").read_text(encoding="utf-8"))
    report["start_behavior"] = start
    report["development"] = {}
    for name, info in agents.items():
        stages = {k: v for k, v in (("start", start), ("short", info["behavior_short"]), ("long", info["behavior"]))
                  if v}
        report["development"][name] = compare(stages, int(config.behavior.n_boot)) if len(stages) > 1 else None
    report["energy_zero"] = {"start": energy_zero_summary(start), **{
        name: {k: energy_zero_summary(info[k]) for k in ("behavior_short", "behavior")} for name, info in agents.items()}}
    (recorder.run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, info in agents.items():
        for ph, v in (info["ethics"].get("phases") or {}).items():
            recorder.log(f"{name} {ph}: energy {v['mean_energy']:.3f}  below 0.1 {v['fraction_below_0.1']:.4f}  "
                         f"below 0.3 {v['fraction_below_0.3']:.4f}  food/1000 {v['food_per_1000_steps']:.2f}  "
                         f"min {v.get('energy_min')}  p10 {v.get('energy_p10')}")
        recorder.log(f"{name} ethics rule: below 0.1 after warm-up "
                     f"{info['ethics'].get('fraction_below_0.1_after_warmup', float('nan')):.4f} "
                     f"(limit {float(config.ethics.max_low_fraction):.2f})")
        z = info["ethics"].get("zero_runs")
        if z:
            recorder.log(f"{name} energy-0 stretches per body: longest {z['longest']} steps, "
                         + "  ".join(f">={t}: {z[f'ge{t}']} ({z[f'ge{t}_after_warmup']} after warm-up)"
                                     for t in (200, 500, 1000)))
            recorder.log(f"{name} individual rule (limit {config.ethics.zero_run_limit} steps): "
                         + (f"stopped the life at iteration {z['flagged_at_iteration']}"
                            if z.get("flagged_at_iteration") is not None else "did not stop the life")
                         + "; candidate thresholds: " + "  ".join(
                             f">={t}: " + (f"iteration {v}" if v is not None else "no")
                             for t, v in z.get("would_flag_at_iteration", {}).items()))
        for key in ("behavior_short", "behavior"):
            rv = battery_review(info.get(key) or {}, config)
            if rv:
                recorder.log(f"{name} REVIEW {key}: episodes {rv} reach {config.ethics.zero_run_limit} steps in a "
                             f"row at energy 0")
        for ph, v in info["drive_shares"].items():
            if v:
                recorder.log(f"{name} {ph} drive shares: " + "  ".join(
                    f"{k} {v['share_abs'][k]:.3f}" for k in COMPONENTS))
    for key in ("behavior_short", "behavior"):
        comp = report[key]
        if not comp:
            continue
        recorder.log(f"--- {key}")
        names = list(comp["means"]["distance"])
        recorder.log("measure            " + "  ".join(f"{n[:24]:>24}" for n in names))
        for m in MEASURES:
            cells = []
            for n in names:
                v = comp["means"][m][n]
                cells.append("n/a" if v is None else f"{v['mean']:.3g} [{v['ci95'][0]:.3g}, {v['ci95'][1]:.3g}]")
            recorder.log(f"{m:<18} " + "  ".join(f"{c:>24}" for c in cells))
        for m in MEASURES:
            for pair, v in comp["hedges_g"][m].items():
                if v is not None:
                    recorder.log(f"g {m:<18} {pair:<60} {v['g']:+.2f} [{v['ci95'][0]}, {v['ci95'][1]}]")
    for name, comp in report["development"].items():
        if not comp:
            continue
        recorder.log(f"--- development {name}")
        for m in ("effort", "distance", "area", "food", "neutral_contacts", "energy_mean"):
            cells = "  ".join(f"{st} {v['mean']:.3g}" for st, v in comp["means"][m].items() if v)
            gs = "  ".join(f"g {pair} {v['g']:+.2f} [{v['ci95'][0]}, {v['ci95'][1]}]"
                           for pair, v in comp["hedges_g"][m].items() if v)
            recorder.log(f"{m:<18} {cells}   {gs}")
    for name, v in report["energy_zero"].items():
        recorder.log(f"energy zero {name}: {v}")
    plot_agents(report, [Path(str(d)) for d in config.report.agents], recorder.run_dir)
