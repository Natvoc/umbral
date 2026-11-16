"""Experiments runnable from the CLI by name.

Each experiment is a function registered with ``@register("name")``. Phase 0 only
ships ``noop`` (smoke test of seeding, recording and checkpointing);
Phase 1 adds ``random_agent`` (a random body living in the world); Phase 2a adds
``bench_env``, ``train_world_model`` and ``eval_world_model``.
"""

from umbral.experiments.registry import available, get_experiment, register
from umbral.experiments import noop as _noop  # noqa: F401  (registers "noop")
from umbral.experiments import random_agent as _random_agent  # noqa: F401  (registers "random_agent")
from umbral.experiments import world_model as _world_model  # noqa: F401  (Phase 2a experiments)

__all__ = ["available", "get_experiment", "register"]
