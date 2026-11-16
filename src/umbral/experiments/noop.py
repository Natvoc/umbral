"""``noop``: smoke test for seeding, recording and checkpointing.

A tiny MLP regresses a noisy linear target on random data. Every step draws from
all three generators (numpy Generator for inputs, torch for noise, Python
``random`` for a masked feature), so any RNG state lost on save/restore changes
the trajectory and breaks the tests.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from umbral.checkpoint import Checkpoint, make_checkpoint, save_checkpoint
from umbral.config import Config
from umbral.experiments.registry import register
from umbral.recorder import Recorder

OnStep = Callable[[int, float], None]


def build(config: Config) -> tuple[nn.Module, torch.optim.Optimizer]:
    c = config.noop
    model = nn.Sequential(nn.Linear(c.in_dim, c.hidden), nn.Tanh(), nn.Linear(c.hidden, 1))
    return model, torch.optim.Adam(model.parameters(), lr=c.lr)


def train_step(config: Config, model: nn.Module, opt: torch.optim.Optimizer,
               np_rng: np.random.Generator) -> float:
    c = config.noop
    x = torch.from_numpy(np_rng.standard_normal((c.batch_size, c.in_dim)).astype(np.float32))
    x[:, random.randrange(c.in_dim)] = 0.0
    y = 0.5 * x.sum(dim=1, keepdim=True) + 0.1 * torch.randn(c.batch_size, 1)
    loss = nn.functional.mse_loss(model(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


def checkpoint_path(run_dir: Path, step: int) -> Path:
    return Path(run_dir) / "checkpoints" / f"step_{step:06d}.pt"


@register("noop")
def run_noop(config: Config, recorder: Recorder, np_rng: np.random.Generator,
             resume: Checkpoint | None) -> None:
    c = config.noop
    model, opt = build(config)
    start = 0
    if resume is not None:
        resume.load_into({"model": model}, {"opt": opt})
        np_rng = resume.activate()  # after build(): model init consumed torch RNG
        start = resume.step
        recorder.log(f"resumed at step {start}")

    for step in range(start + 1, c.steps + 1):
        loss = train_step(config, model, opt, np_rng)
        recorder.log_metric(step, "loss", loss)
        if step == c.checkpoint_at:
            ckpt = make_checkpoint(step, {"model": model}, {"opt": opt}, np_rng, config)
            path = save_checkpoint(checkpoint_path(recorder.run_dir, step), ckpt)
            recorder.log(f"checkpoint saved: {path}")
        if step % 50 == 0 or step == c.steps:
            recorder.log(f"step {step}/{c.steps} loss={loss:.6f}")
