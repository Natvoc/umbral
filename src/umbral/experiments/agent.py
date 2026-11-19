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
from umbral.mind.evaluation import bootstrap_ratio_ci, evaluate
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


def prediction_check(world, baseline_path: str | Path, config: Config) -> dict[str, Any]:
    """The agent's self-model vs the childhood model, Phase 2a protocol (OU policy, eval seeds).

    Both are evaluated on the same environments, in the *current* world physics, so the
    baseline is always measured in the same world as the agent. Ratio agent / childhood
    per channel with a 95 % bootstrap CI over seeds (stratified by history).
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
    sums = evaluate({"agent": world, "childhood": w0}, config, histories, [EVAL_SEED_BASE + i for i in range(n)],
                    [EVAL_SEED_BASE + POLICY_SEED_OFFSET + i for i in range(n)], int(ec.steps), int(ec.chunk))
    rows = sums["rows"]
    out: dict[str, Any] = {"baseline": str(baseline_path), "envs": n, "steps": int(ec.steps), "channels": {}}
    for c in CHANNELS:
        a, b = rows["mse"]["agent"][c], rows["mse"]["childhood"][c]
        ma, mb = a[:, 0].sum() / a[:, 1].sum(), b[:, 0].sum() / b[:, 1].sum()
        lo, hi = bootstrap_ratio_ci(a, b, histories, int(ec.n_boot), int(ec.bootstrap_seed))
        out["channels"][c] = {"agent": float(ma), "childhood": float(mb), "ratio": float(ma / mb),
                              "ci95": [lo, hi]}
    return out


@register("live_agent")
def run_live_agent(config: Config, recorder: Recorder, np_rng: np.random.Generator,
                   resume: Checkpoint | None) -> None:
    life = Life(config, recorder, np_rng, resume)
    summary = life.run()
    d = recorder.run_dir
    if summary["life_status"] == "flagged":
        (d / "flagged_summary.json").write_text(json.dumps(summary["ethics"], indent=2) + "\n", encoding="utf-8")
        recorder.log(f"FLAGGED by the ethics rule: {summary['ethics']['fraction_below_0.1_after_warmup']:.3f} "
                     f"of the life below {config.ethics.low_energy} after warm-up")
        return
    recorder.log(f"life {summary['life_status']}: {summary['env_steps_per_second']} steps/s, "
                 f"time fractions {summary['time_fractions']}")
    if summary["life_status"] == "completed" and bool(config.life.run_report):
        agent = life.agent
        agent.eval()
        beh = run_battery(agent, config, str(config.agent.history))
        (d / "behavior.json").write_text(json.dumps(beh, indent=2) + "\n", encoding="utf-8")
        short = find_checkpoint(d, "life_short")
        if short is not None:
            agent_s, _ = load_agent(load_checkpoint(short))
            beh_s = run_battery(agent_s, config, str(config.agent.history))
            beh_s["checkpoint"] = str(short)
            (d / "behavior_short.json").write_text(json.dumps(beh_s, indent=2) + "\n", encoding="utf-8")
        pc = prediction_check(agent.world, config.agent.infancy, config)
        (d / "prediction_check.json").write_text(json.dumps(pc, indent=2) + "\n", encoding="utf-8")
        for c, v in pc["channels"].items():
            recorder.log(f"prediction {c}: agent/childhood {v['ratio']:.3f} "
                         f"[{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")


@register("behavior_battery")
def run_behavior_battery(config: Config, recorder: Recorder, np_rng: np.random.Generator,
                         resume: Checkpoint | None) -> None:
    if not config.behavior.checkpoint:
        raise ValueError("behavior.checkpoint must point to a live_agent checkpoint")
    ckpt = load_checkpoint(Path(str(config.behavior.checkpoint)))
    agent, acfg = load_agent(ckpt)
    beh = run_battery(agent, config, str(acfg.agent.history))
    beh["checkpoint"] = str(config.behavior.checkpoint)
    recorder.update_meta(temperament=str(acfg.agent.temperament), history=str(acfg.agent.history),
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
    (recorder.run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, info in agents.items():
        for ph, v in (info["ethics"].get("phases") or {}).items():
            recorder.log(f"{name} {ph}: energy {v['mean_energy']:.3f}  below 0.1 {v['fraction_below_0.1']:.4f}  "
                         f"below 0.3 {v['fraction_below_0.3']:.4f}  food/1000 {v['food_per_1000_steps']:.2f}  "
                         f"min {v.get('energy_min')}  p10 {v.get('energy_p10')}")
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
    plot_agents(report, [Path(str(d)) for d in config.report.agents], recorder.run_dir)
