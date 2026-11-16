"""Phase 2a experiments: ``bench_env``, ``train_world_model``, ``eval_world_model``.

- ``bench_env``: environment-steps per second of ``BatchedUmbralEnv`` for each
  ``bench.sizes`` (metrics ``env_steps_per_second`` at step N; also in meta.json).
- ``train_world_model``: trains the efference and no-efference models side by
  side (see ``umbral.mind.training``); resumable with ``run.resume_from``.
- ``eval_world_model``: on unseen seeds in every history, compares the per-channel
  prediction error of the efference model, the no-efference model and
  persistence; writes ``eval_results.json`` and ``learning_curves.png``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from umbral.checkpoint import Checkpoint, load_checkpoint
from umbral.config import Config
from umbral.experiments.registry import register
from umbral.mind.evaluation import TOUCH_EVENTS, evaluate, summarize
from umbral.mind.model import CHANNELS, build_model
from umbral.mind.policy import OUPolicy
from umbral.mind.report import plot_checkpoint_eval, plot_learning_curves
from umbral.mind.training import (EVAL_SEED_BASE, MODEL_NAMES, POLICY_SEED_OFFSET, Trainer, history_cycle,
                                  peak_memory_mb)
from umbral.recorder import Recorder
from umbral.world.batched import BatchedUmbralEnv


@register("bench_env")
def run_bench_env(config: Config, recorder: Recorder, np_rng: np.random.Generator,
                  resume: Checkpoint | None) -> None:
    results = {}
    for n in config.bench.sizes:
        n = int(n)
        env = BatchedUmbralEnv(config, n, history_cycle(config, n))
        env.reset(seed=[int(np_rng.integers(0, 2**31)) for _ in range(n)])
        policy = OUPolicy(n, float(config.train.smoothing), float(config.train.scale),
                          [int(np_rng.integers(0, 2**31)) for _ in range(n)])
        for _ in range(20):
            env.step(policy.act())
        steps, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < float(config.bench.seconds):
            env.step(policy.act())
            steps += 1
        sps = steps * n / (time.perf_counter() - t0)
        results[str(n)] = round(sps, 1)
        recorder.log(f"N={n}: {sps:.0f} env-steps/s")
    # Timings vary run to run, so they go to meta.json and the log, not to metrics.jsonl.
    recorder.update_meta(env_steps_per_second=results)


@register("train_world_model")
def run_train_world_model(config: Config, recorder: Recorder, np_rng: np.random.Generator,
                          resume: Checkpoint | None) -> None:
    trainer = Trainer(config, recorder, np_rng, resume)
    summary = trainer.run()
    recorder.log(f"time fractions: {summary['time_fractions']}  peak memory: {summary['peak_memory_mb']} MB")
    order = plot_learning_curves(recorder.run_dir / "metrics.jsonl", recorder.run_dir / "learning_curves.png")
    plot_checkpoint_eval(recorder.run_dir / "metrics.jsonl", recorder.run_dir / "checkpoint_eval_curves.png")
    recorder.update_meta(learning_order=order)


@register("eval_world_model")
def run_eval_world_model(config: Config, recorder: Recorder, np_rng: np.random.Generator,
                         resume: Checkpoint | None) -> None:
    ec = config.eval
    if not ec.checkpoint:
        raise ValueError("eval.checkpoint must point to a train_world_model checkpoint")
    ckpt_path = Path(ec.checkpoint)
    ckpt = load_checkpoint(ckpt_path)
    tcfg = Config(ckpt.config)  # world, body and mind exactly as trained
    hids = sorted(tcfg.world.histories)
    histories = [h for h in hids for _ in range(int(ec.envs_per_history))]
    n = len(histories)
    models = load_models(ckpt, tcfg)
    sums = evaluate(models, tcfg, histories, [EVAL_SEED_BASE + i for i in range(n)],
                    [EVAL_SEED_BASE + POLICY_SEED_OFFSET + i for i in range(n)], int(ec.steps), int(ec.chunk))
    res = summarize(sums)
    table, events = res["mse"], res["touch_events"]
    for c in CHANNELS:
        for p in table[c]:
            recorder.log_metric(0, f"{p}/mse_{c}", table[c][p])
    for e in TOUCH_EVENTS:
        for p, v in events[e].items():
            if v is not None:
                recorder.log_metric(0, f"{p}/touch_{e}", v)

    recorder.log(f"{'channel':<9} {'efference':>12} {'no_efference':>13} {'persistence':>12} "
                 f"{'eff/pers':>9} {'eff/noeff':>10}")
    for c in CHANNELS:
        t = table[c]
        recorder.log(f"{c:<9} {t['efference']:>12.4e} {t['no_efference']:>13.4e} {t['persistence']:>12.4e} "
                     f"{t['efference'] / t['persistence']:>9.3f} {t['efference'] / t['no_efference']:>10.3f}")
    recorder.log(f"{'touch event':<16} {'n':>7} {'efference':>12} {'no_efference':>13} {'eff/noeff':>10}")
    for e in TOUCH_EVENTS:
        v, cnt = events[e], res["touch_event_counts"][e]
        ratio = (v["efference"] / v["no_efference"]) if cnt and v["no_efference"] else float("nan")
        fmt = (lambda x: f"{x:.4e}" if x is not None else "n/a")
        recorder.log(f"{e:<16} {cnt:>7} {fmt(v['efference']):>12} {fmt(v['no_efference']):>13} {ratio:>10.3f}")

    ratio = {c: table[c]["efference"] / table[c]["persistence"] for c in CHANNELS}
    ev = events["onset_self"]
    criteria = {
        "efference_vs_persistence_ratio": ratio,
        "beats_persistence_all_channels": all(r < 1.0 for r in ratio.values()),
        "beats_no_efference": {c: table[c]["efference"] < table[c]["no_efference"] for c in CHANNELS},
        "touch_onset_self_beats_no_efference": (ev["efference"] is not None and ev["no_efference"] is not None
                                                and ev["efference"] < ev["no_efference"]),
    }
    results = {"checkpoint": str(ckpt_path), "trained_updates": ckpt.step, "eval_steps": int(ec.steps),
               "envs": n, "histories": hids, **res, "criteria": criteria}
    train_metrics = ckpt_path.resolve().parents[1] / "metrics.jsonl"
    if train_metrics.is_file():
        results["learning_order"] = plot_learning_curves(train_metrics, recorder.run_dir / "learning_curves.png")
        plot_checkpoint_eval(train_metrics, recorder.run_dir / "checkpoint_eval_curves.png")
    (recorder.run_dir / "eval_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    recorder.update_meta(peak_memory_mb=peak_memory_mb(), criteria=criteria)


def load_models(ckpt: Checkpoint, tcfg: Config) -> dict:
    from umbral.body.sensors import ObsLayout

    layout = ObsLayout(int(tcfg.body.vision_rays))
    models = {k: build_model(layout, tcfg.mind, efference_copy=(k == "efference")) for k in MODEL_NAMES}
    for k, m in models.items():
        m.load_state_dict(ckpt.models[k])
        m.eval()
    return models
