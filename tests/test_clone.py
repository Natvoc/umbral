import json

import torch

from conftest import BASE_CONFIG, read_metrics
from umbral.checkpoint import clone, load_checkpoint
from umbral.config import load_config
from umbral.experiments.noop import build, checkpoint_path, train_step


def assert_same_state(a, b):
    """Recursively compare nested dicts/lists/tensors/scalars for exact equality."""
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and torch.equal(a, b)
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys()
        for k in a:
            assert_same_state(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            assert_same_state(x, y)
    else:
        assert a == b


def test_clone_gives_identical_independent_copies(run_noop):
    ckpt = load_checkpoint(checkpoint_path(run_noop("seed=5"), 100))
    copies = clone(ckpt, 3)
    assert len(copies) == 3
    for c in copies:
        assert c is not ckpt
        assert c.step == ckpt.step
        assert_same_state(c.models, ckpt.models)
        assert_same_state(c.optimizers, ckpt.optimizers)
        assert_same_state(c.rng, ckpt.rng)
        assert c.config == ckpt.config

    # Independence: mutating one copy touches neither the source nor its siblings.
    weight = next(iter(copies[0].models["model"].values()))
    weight.add_(1.0)
    assert not torch.equal(weight, next(iter(ckpt.models["model"].values())))
    assert torch.equal(next(iter(copies[1].models["model"].values())),
                       next(iter(ckpt.models["model"].values())))


def run_copy(config, copy, steps):
    model, opt = build(config)
    copy.load_into({"model": model}, {"opt": opt})
    np_rng = copy.activate()
    return [train_step(config, model, opt, np_rng) for _ in range(steps)]


def test_sequential_copies_share_trajectory(run_noop):
    """Two copies run back to back in one process produce the same trajectory,
    equal to the continuous run's second half."""
    full = run_noop("seed=11")
    config = load_config(BASE_CONFIG)
    a, b = clone(checkpoint_path(full, 100), 2)

    traj_a = run_copy(config, a, 100)
    traj_b = run_copy(config, b, 100)  # global RNG was advanced by copy a; activate() resets it

    assert traj_a == traj_b
    continuous = [json.loads(l)["value"] for l in read_metrics(full) if json.loads(l)["step"] > 100]
    assert traj_a == continuous
