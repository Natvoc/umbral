"""``train_population``: the queue of agents (temperament x history x seed), resumable.

Everything lives in one fixed folder, ``population.dir``:

- ``config.yaml``: the configuration frozen when the population was created. Every
  agent of every session runs with it, so later sessions cannot drift.
- ``population.json``: the index, one entry per agent with its temperament,
  history, seed, ``status`` (``pending``, ``partial``, ``done`` or ``flagged``),
  the run folders it used and the path of its ``life_long.pt``.
- ``agents/``: the agents' run folders (``live_agent`` runs).

Each agent runs as a separate process (``python -m umbral run``). A session runs
the pending and partial agents in order, at most ``population.max_agents`` of them,
``population.parallel`` at a time (threads split among them). It skips the agents
that are done. A partial agent (interrupted, or paused by ``life.stop_at``) resumes
from its latest checkpoint. If an agent is flagged by the ethics rule, the whole
queue stops. The flagged agent leaves ``flagged_summary.json`` for review before
resuming.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from umbral.checkpoint import Checkpoint
from umbral.config import Config
from umbral.experiments.registry import register
from umbral.recorder import Recorder


def agent_id(temperament: str, history: str, seed_index: int) -> str:
    return f"{temperament}_h{history}_s{seed_index}"


def agent_seed(t_idx: int, h_idx: int, s_idx: int) -> int:
    """Deterministic, distinct seed per agent (training range)."""
    return 1000 * (s_idx + 1) + 10 * t_idx + h_idx


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _latest_checkpoint(runs: list[str]) -> Path | None:
    files = [f for r in runs for f in (Path(r) / "checkpoints").glob("*.pt")]
    return max(files, key=lambda f: f.stat().st_mtime_ns) if files else None


def init_population(config: Config, pop_dir: Path) -> dict[str, Any]:
    pc = config.population
    pop_dir.mkdir(parents=True, exist_ok=True)
    (pop_dir / "agents").mkdir(exist_ok=True)
    cfg_path = pop_dir / "config.yaml"
    if not cfg_path.exists():
        cfg_path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True),
                            encoding="utf-8")
    index_path = pop_dir / "population.json"
    if index_path.exists():
        return json.loads(index_path.read_text(encoding="utf-8"))
    agents = {}
    for t_idx, temp in enumerate(pc.temperaments):
        for h_idx, hist in enumerate(pc.histories):
            for s_idx in pc.seeds:
                aid = agent_id(str(temp), str(hist), int(s_idx))
                agents[aid] = {"temperament": str(temp), "history": str(hist), "seed_index": int(s_idx),
                               "seed": agent_seed(t_idx, h_idx, int(s_idx)), "status": "pending", "runs": [],
                               "life_long": None}
    index = {"created_with_overrides": list(pc.overrides), "agents": agents}
    _write_json(index_path, index)
    return index


def _command(pop_dir: Path, entry: dict[str, Any], aid: str, threads: int, overrides: list[str]) -> list[str]:
    cmd = [sys.executable, "-m", "umbral", "run", "--config", str(pop_dir / "config.yaml"),
           "experiment=live_agent", f"run.output_dir={(pop_dir / 'agents').as_posix()}", f"run.name={aid}",
           f"seed={entry['seed']}", f"agent.temperament={entry['temperament']}",
           f"agent.history={entry['history']}", f"torch.num_threads={threads}", *overrides]
    ckpt = _latest_checkpoint(entry["runs"]) if entry["status"] == "partial" else None
    if ckpt is not None:
        cmd.append(f"run.resume_from={ckpt.as_posix()}")
    return cmd


def _finish(entry: dict[str, Any], run_dir: Path | None) -> None:
    if run_dir is None:
        entry["status"] = "partial"
        return
    entry["runs"].append(str(run_dir))
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    life = meta.get("life_status")
    if life == "completed" and (run_dir / "checkpoints" / "life_long.pt").is_file():
        entry["status"] = "done"
        entry["life_long"] = str(run_dir / "checkpoints" / "life_long.pt")
        if meta.get("battery_review"):  # individual rule in the battery: the agent waits for review
            entry["review"] = meta["battery_review"]
    elif life == "flagged":
        entry["status"] = "flagged"
    else:
        entry["status"] = "partial"


@register("train_population")
def run_train_population(config: Config, recorder: Recorder, np_rng: np.random.Generator,
                         resume: Checkpoint | None) -> None:
    pc = config.population
    pop_dir = Path(str(pc.dir))
    index = init_population(config, pop_dir)
    index_path = pop_dir / "population.json"
    agents = index["agents"]
    for e in agents.values():  # a session that died mid-agent leaves it "running"
        if e["status"] == "running":
            e["status"] = "partial"
    if any(e["status"] == "flagged" for e in agents.values()):
        recorder.log("a flagged agent is waiting for review: the queue does not start")
        recorder.update_meta(population_stopped="flagged agent pending review")
        _write_json(index_path, index)
        return
    queue = [a for a, e in agents.items() if e["status"] in ("pending", "partial")]
    if pc.max_agents is not None:
        queue = queue[: int(pc.max_agents)]
    parallel = max(1, int(pc.parallel))
    threads = max(1, int(config.torch.num_threads) // parallel)
    overrides = [str(o) for o in index.get("created_with_overrides", [])]
    recorder.log(f"population at {pop_dir}: {len(queue)} agents this session, {parallel} at a time, "
                 f"{threads} threads each")
    running: dict[str, tuple[subprocess.Popen, float, tuple]] = {}
    started = {"n": 0}
    timings: dict[str, float] = {}
    stop = False

    def launch(aid: str) -> None:
        e = agents[aid]
        cmd = _command(pop_dir, e, aid, threads, overrides)
        log_path = pop_dir / "agents" / f"{aid}.out.txt"
        log = open(log_path, "a", encoding="utf-8")
        log.write(f"--- session {recorder.run_dir.name}\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
        running[aid] = (proc, time.perf_counter(), (log, log_path, log_path.stat().st_size))
        e["status"] = "running"
        _write_json(index_path, index)
        recorder.log(f"start {aid}" + ("  (resume)" if "run.resume_from" in " ".join(cmd) else ""))

    wall0 = time.perf_counter()
    while (queue and not stop) or running:
        while queue and not stop and len(running) < parallel:
            launch(queue.pop(0))
            started["n"] += 1
        time.sleep(0.2)
        for aid in list(running):
            proc, t0, (log, log_path, offset) = running[aid]
            if proc.poll() is None:
                continue
            log.close()
            with open(log_path, encoding="utf-8", errors="replace") as f:
                f.seek(offset)
                out = f.read()
            del running[aid]
            run_dir = None
            for line in out.splitlines():
                if line.startswith("done: "):
                    run_dir = Path(line[len("done: "):].strip())
            _finish(agents[aid], run_dir)
            timings[aid] = round(time.perf_counter() - t0, 1)
            _write_json(index_path, index)
            recorder.log(f"end {aid}: {agents[aid]['status']} in {timings[aid] / 60:.1f} min")
            if agents[aid]["status"] == "flagged":
                stop = True
                recorder.log(f"{aid} FLAGGED by the ethics rule: the queue stops (see its flagged_summary.json)")
    counts = {s: sum(e["status"] == s for e in agents.values()) for s in ("pending", "partial", "done", "flagged")}
    recorder.update_meta(population_dir=str(pop_dir), agent_seconds=timings, counts=counts,
                         session_seconds=round(time.perf_counter() - wall0, 1), parallel=parallel,
                         threads_per_agent=threads)
    recorder.log(f"population: {counts}")
