"""Experiments runnable from the CLI by name.

Each experiment is a function registered with ``@register("name")``. Phase 0 only
ships ``noop``, a smoke test of seeding, recording and checkpointing.
"""

from umbral.experiments.registry import available, get_experiment, register
from umbral.experiments import noop as _noop  # noqa: F401  (registers "noop")

__all__ = ["available", "get_experiment", "register"]
