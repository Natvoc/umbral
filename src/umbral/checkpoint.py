"""Full-state checkpoints: models, optimizers and every random generator.

Guarantees:

- Resuming from a checkpoint reproduces exactly the trajectory of a run that
  never stopped.
- ``clone(checkpoint, n)`` returns ``n`` independent, identical copies of the
  agent at that point (generators included), so any later difference between
  copies comes only from the experimental condition each one receives.

Format v2 (Phase 2a) adds the agent's world: ``env`` (the full state of a
``BatchedUmbralEnv``: bodies, arenas, food timers, world and link generators,
BodyLink modes, fade counters and frozen noise statistics), ``policy`` (the
action policy's state, e.g. OU noise and its generators) and ``mind_state``
(recurrent hidden states and anything else the mind carries between steps).
``env`` and ``policy`` hold numpy arrays in memory and are stored as tensors;
``mind_state`` holds tensors. v1 files still load (those fields are None).

``random`` and ``torch`` keep their random state globally, per process. Before a
copy runs, call its ``activate()``: it restores *that copy's* generators into the
global state and returns its numpy ``Generator``. Run copies one after another,
each calling ``activate()`` right before it starts.

``activate()`` restores the state stored in the checkpoint (the copy's starting
point). It does not track a copy's progress; to pause a copy midway and resume it
later, save a new checkpoint from it.
"""

from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

FORMAT_VERSION = 2
SUPPORTED_VERSIONS = (1, 2)


# --- RNG state <-> plain data -------------------------------------------------
# Non-tensor RNG state is stored as JSON strings so files load with
# torch.load(weights_only=True) (no arbitrary code on load).

def capture_rng_state(np_rng: np.random.Generator | None) -> dict[str, Any]:
    py_version, py_internal, py_gauss = random.getstate()
    name, keys, pos, has_gauss, cached = np.random.get_state(legacy=True)
    return {
        "python": json.dumps([py_version, list(py_internal), py_gauss]),
        "numpy_global": json.dumps({
            "name": name, "keys": keys.tolist(), "pos": int(pos),
            "has_gauss": int(has_gauss), "cached_gaussian": float(cached),
        }),
        "numpy_generator": None if np_rng is None else json.dumps(np_rng.bit_generator.state),
        "torch": torch.get_rng_state().clone(),
    }


def restore_rng_state(state: Mapping[str, Any]) -> np.random.Generator | None:
    """Restore global generators from ``state``; return a fresh numpy Generator (or None)."""
    py_version, py_internal, py_gauss = json.loads(state["python"])
    random.setstate((py_version, tuple(py_internal), py_gauss))
    g = json.loads(state["numpy_global"])
    np.random.set_state((g["name"], np.array(g["keys"], dtype=np.uint32), g["pos"],
                         g["has_gauss"], g["cached_gaussian"]))
    torch.set_rng_state(state["torch"].clone())
    if state["numpy_generator"] is None:
        return None
    bg_state = json.loads(state["numpy_generator"])
    bit_generator = getattr(np.random, bg_state["bit_generator"])()
    bit_generator.state = bg_state
    return np.random.Generator(bit_generator)


# --- Checkpoint ---------------------------------------------------------------

@dataclass
class Checkpoint:
    step: int
    models: dict[str, dict[str, Any]]
    optimizers: dict[str, dict[str, Any]]
    rng: dict[str, Any]
    config: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    env: dict[str, Any] | None = None
    policy: dict[str, Any] | None = None
    mind_state: dict[str, Any] | None = None

    def restore_env(self, env: Any) -> None:
        """Load this checkpoint's world into an environment built from the same config."""
        if self.env is None:
            raise ValueError("checkpoint has no environment state")
        env.set_state(copy.deepcopy(self.env))

    def load_into(
        self,
        models: Mapping[str, torch.nn.Module],
        optimizers: Mapping[str, torch.optim.Optimizer] | None = None,
    ) -> None:
        """Load weights and optimizer state into freshly built objects.

        State is deep-copied first: optimizers may otherwise alias the
        checkpoint's tensors and mutate them in place while training.
        """
        optimizers = optimizers or {}
        _check_keys("models", self.models, models)
        _check_keys("optimizers", self.optimizers, optimizers)
        for name, module in models.items():
            module.load_state_dict(copy.deepcopy(self.models[name]))
        for name, opt in optimizers.items():
            opt.load_state_dict(copy.deepcopy(self.optimizers[name]))

    def activate(self) -> np.random.Generator | None:
        """Restore this checkpoint's generators into the global state.

        Call right before running, after building/loading models (model
        construction consumes torch randomness). Returns the numpy Generator.
        """
        return restore_rng_state(self.rng)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "step": self.step,
            "models": self.models,
            "optimizers": self.optimizers,
            "rng": self.rng,
            "config": self.config,
            "extra": self.extra,
            "env": _encode(self.env),
            "policy": _encode(self.policy),
            "mind_state": self.mind_state,
        }


def _encode(value: Any) -> Any:
    """numpy -> torch (for weights_only-safe storage), recursively."""
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(value).copy())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    return value


def _decode(value: Any) -> Any:
    """torch -> numpy, recursively (inverse of ``_encode``)."""
    if isinstance(value, torch.Tensor):
        return value.numpy().copy()
    if isinstance(value, dict):
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


def _check_keys(kind: str, stored: Mapping[str, Any], given: Mapping[str, Any]) -> None:
    if set(stored) != set(given):
        raise KeyError(f"{kind}: checkpoint has {sorted(stored)}, got {sorted(given)}")


def make_checkpoint(
    step: int,
    models: Mapping[str, torch.nn.Module],
    optimizers: Mapping[str, torch.optim.Optimizer],
    np_rng: np.random.Generator | None,
    config: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    env: Any = None,
    policy: Any = None,
    mind_state: Mapping[str, Any] | None = None,
) -> Checkpoint:
    """Snapshot live objects and the current global RNG state (deep copies).

    ``env`` and ``policy`` are live objects with ``get_state()``; ``mind_state``
    is a dict of tensors (cloned).
    """
    to_plain = getattr(config, "to_dict", None)
    return Checkpoint(
        step=int(step),
        models={k: copy.deepcopy(m.state_dict()) for k, m in models.items()},
        optimizers={k: copy.deepcopy(o.state_dict()) for k, o in optimizers.items()},
        rng=capture_rng_state(np_rng),
        config=to_plain() if to_plain else dict(config or {}),
        extra=copy.deepcopy(dict(extra or {})),
        env=None if env is None else copy.deepcopy(env.get_state()),
        policy=None if policy is None else copy.deepcopy(policy.get_state()),
        mind_state=None if mind_state is None else {
            k: (v.detach().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v))
            for k, v in mind_state.items()},
    )


def save_checkpoint(path: str | Path, checkpoint: Checkpoint) -> Path:
    """Write atomically (temp file + rename) so a crash never leaves a half file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(checkpoint.to_dict(), tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | Path) -> Checkpoint:
    data = torch.load(Path(path), map_location="cpu", weights_only=True)
    version = data.get("format_version")
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"{path}: unsupported checkpoint format {version!r}")
    return Checkpoint(
        step=data["step"], models=data["models"], optimizers=data["optimizers"],
        rng=data["rng"], config=data["config"], extra=data["extra"],
        env=_decode(data.get("env")), policy=_decode(data.get("policy")),
        mind_state=data.get("mind_state"),
    )


def clone(checkpoint: Checkpoint | str | Path, n: int) -> list[Checkpoint]:
    """Return ``n`` independent, identical copies of the agent at ``checkpoint``.

    Each copy owns its tensors, RNG state and (v2) its world, policy and mind
    state (no sharing with the source or with each other). Run each one after
    calling its ``activate()``; restore its world with ``restore_env(env)``.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    source = checkpoint if isinstance(checkpoint, Checkpoint) else load_checkpoint(checkpoint)
    return [copy.deepcopy(source) for _ in range(n)]
