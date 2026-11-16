"""Command line: ``umbral run --config configs/base.yaml [key=value ...]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from umbral.checkpoint import load_checkpoint
from umbral.config import Config, load_config
from umbral.experiments import available, get_experiment
from umbral.recorder import Recorder
from umbral.seeding import seed_everything


def run_experiment(config: Config) -> Path:
    """Seed, open a recorder, run the configured experiment; return the run directory."""
    experiment = get_experiment(config.experiment)
    seeded = seed_everything(config.seed, config.torch.num_threads, config.torch.num_interop_threads)
    resume_from = config.run.resume_from
    resume = load_checkpoint(resume_from) if resume_from else None
    meta = {
        "threads": seeded.threads.as_dict(),
        "resumed_from": None if resume is None else str(resume_from),
        "resumed_at_step": None if resume is None else resume.step,
        # On resume the checkpoint's RNG state replaces the one derived from config.seed.
        "seed_overridden_by_checkpoint": resume is not None,
    }
    with Recorder(config, meta=meta) as recorder:
        experiment(config, recorder, seeded.np_rng, resume)
    return recorder.run_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="umbral")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run an experiment")
    run.add_argument("--config", required=True, help="base YAML config")
    run.add_argument("overrides", nargs="*", help="key=value overrides, e.g. seed=3 run.name=prueba")
    sub.add_parser("list", help="list available experiments")
    args = parser.parse_args(argv)

    if args.command == "list":
        print("\n".join(available()))
        return 0
    try:
        config = load_config(args.config, args.overrides)
    except (KeyError, ValueError, OSError) as e:
        print(f"umbral: config error: {e}", file=sys.stderr)
        return 2
    run_dir = run_experiment(config)
    print(f"done: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
