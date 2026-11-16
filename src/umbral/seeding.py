"""Reproducible seeding and CPU thread limits.

Project rule: new code draws numpy randomness from the explicit
``np.random.Generator`` returned here, never from the global ``np.random.*``
functions. The global numpy state is still seeded and checkpointed as a safety
net for third-party code.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

_interop_configured = False


@dataclass(frozen=True)
class ThreadInfo:
    num_threads: int
    num_interop_threads: int

    def as_dict(self) -> dict[str, int]:
        return {"num_threads": self.num_threads, "num_interop_threads": self.num_interop_threads}


@dataclass(frozen=True)
class Seeded:
    seed: int
    np_rng: np.random.Generator
    threads: ThreadInfo


def configure_threads(num_threads: int, num_interop_threads: int) -> ThreadInfo:
    """Set torch thread counts and return the values actually in effect.

    ``set_num_threads`` may be called any time. ``set_num_interop_threads`` can
    only be called once per process, before any inter-op parallel work, so it is
    applied only on the first call; later calls (e.g. several runs inside one
    pytest process) keep whatever value is already active.
    """
    global _interop_configured
    torch.set_num_threads(int(num_threads))
    if not _interop_configured:
        _interop_configured = True
        try:
            torch.set_num_interop_threads(int(num_interop_threads))
        except RuntimeError:
            pass  # parallel work already started in this process; keep current value
    return ThreadInfo(torch.get_num_threads(), torch.get_num_interop_threads())


def seed_everything(seed: int, num_threads: int = 4, num_interop_threads: int = 1) -> Seeded:
    """Seed ``random``, global numpy and torch; make torch deterministic on CPU."""
    seed = int(seed)
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    threads = configure_threads(num_threads, num_interop_threads)
    return Seeded(seed=seed, np_rng=np.random.default_rng(seed), threads=threads)
