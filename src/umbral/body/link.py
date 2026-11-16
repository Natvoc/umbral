"""BodyLink: the only path between body and mind, modifiable per channel and part.

Every observation the mind receives passes through ``sense`` and every action it
sends passes through ``act``. Signals are split into segments
``(channel, part)``: sensory channels ``proprio``, ``touch``, ``vision``,
``intero`` and the ``motor`` channel; parts ``trunk``, ``left_limb``,
``right_limb`` (see ``umbral.body.sensors`` for which indices belong where).

Modes, settable per segment with ``set_mode(mode, channel=..., part=...)``:

- ``intact``: the signal passes unchanged.
- ``cut``: the signal is replaced by zeros.
- ``noise`` (sensory only): the signal is replaced by i.i.d. Gaussian noise with
  the per-element mean and standard deviation of the *embodied* signal. These
  statistics are accumulated only while a segment is ``intact`` and are frozen at
  the moment ``noise`` is activated; the body's signal computed afterwards never
  feeds them.
- ``fade``: the signal is multiplied by ``max(0, 1 - k / steps)``, where ``k``
  is the number of steps since the mode was set (first faded step: ``1 - 1/steps``;
  zero from step ``steps`` on).

Motor side: a disconnected part does not move (``cut``; ``fade`` attenuates its
commands), but the action the mind intended is always kept. The difference
between intended and applied commands is the "void action", a central metric.

Timing: the environment calls ``tick()`` once at the start of each step, then
``act`` and ``sense``. Noise draws use each environment's own link generator, so
they never change the world's random stream.

``BatchedBodyLink`` holds one link per environment as arrays (B, ...);
``BodyLink`` is one row of it (a standalone ``BodyLink`` owns a batch of one).
"""

from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np

from umbral.body.sensors import CHANNELS, MOTOR_SEGMENTS, PARTS, ObsLayout, Segment

MODES = ("intact", "cut", "noise", "fade")
MOTOR = "motor"
INTACT, CUT, NOISE, FADE = range(4)


class BatchedBodyLink:
    def __init__(self, layout: ObsLayout, rngs: Sequence[np.random.Generator]):
        self.layout = layout
        self.rngs = list(rngs)
        B, D = len(self.rngs), layout.dim
        self.n = B
        self.segments: list[Segment] = [*layout.segments(), *MOTOR_SEGMENTS]
        self.n_sensory = len(layout.segments())
        S = len(self.segments)
        self.t = np.zeros(B, np.int64)
        self.mode = np.zeros((B, S), np.int64)
        self.t0 = np.zeros((B, S), np.int64)
        self.steps = np.zeros((B, S), np.int64)
        self.noise_mean = np.zeros((B, D))
        self.noise_std = np.zeros((B, D))
        self.count = np.zeros((B, D))
        self.mean = np.zeros((B, D))
        self.m2 = np.zeros((B, D))
        self.intact_mask = np.ones((B, D), bool)

    # --- configuration -------------------------------------------------------

    def _select(self, channel: str | None, part: str | None) -> list[int]:
        if channel is not None and channel not in (*CHANNELS, MOTOR):
            raise ValueError(f"unknown channel {channel!r}")
        if part is not None and part not in PARTS:
            raise ValueError(f"unknown part {part!r}")
        segs = [k for k, s in enumerate(self.segments)
                if (channel is None or s.channel == channel) and (part is None or s.part == part)]
        if not segs:
            raise ValueError(f"no segment for channel={channel!r}, part={part!r}")
        return segs

    def set_mode(self, mode: str, channel: str | None = None, part: str | None = None,
                 steps: int | None = None, envs: Sequence[int] | None = None) -> None:
        """Set ``mode`` on matching segments of the given environments (None = all)."""
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        if mode == "fade" and (steps is None or steps < 1):
            raise ValueError("fade needs steps >= 1")
        segs = self._select(channel, part)
        rows = np.arange(self.n) if envs is None else np.asarray(envs, int)
        code = MODES.index(mode)
        if code == NOISE:
            for k in segs:
                s = self.segments[k]
                if s.channel == MOTOR:
                    raise ValueError("noise is defined only for sensory channels")
                if (self.count[np.ix_(rows, s.idx)] < 2).any():
                    raise ValueError(f"no embodied statistics yet for {s.channel}/{s.part}")
        for k in segs:
            idx = self.segments[k].idx
            self.mode[rows, k] = code
            self.t0[rows, k] = self.t[rows]
            self.steps[rows, k] = int(steps or 0)
            if code == NOISE:  # freeze embodied statistics now
                self.noise_mean[np.ix_(rows, idx)] = self.mean[np.ix_(rows, idx)]
                self.noise_std[np.ix_(rows, idx)] = self._std(rows, idx)
        self._refresh_mask(rows)

    def _refresh_mask(self, rows: np.ndarray) -> None:
        for r in rows:
            self.intact_mask[r] = False
            for k in range(self.n_sensory):
                if self.mode[r, k] == INTACT:
                    self.intact_mask[r, self.segments[k].idx] = True

    def _std(self, rows: np.ndarray, idx: np.ndarray) -> np.ndarray:
        sub = np.ix_(rows, idx)
        return np.sqrt(self.m2[sub] / np.maximum(self.count[sub], 1))

    def reset_rows(self, rows: Sequence[int]) -> None:
        """Fresh link (intact, no statistics, t = 0) for the given environments."""
        rows = np.asarray(rows, int)
        for arr in (self.t, self.mode, self.t0, self.steps, self.noise_mean, self.noise_std,
                    self.count, self.mean, self.m2):
            arr[rows] = 0
        self.intact_mask[rows] = True

    def modes(self, row: int) -> dict[str, str]:
        return {f"{s.channel}/{s.part}": MODES[self.mode[row, k]] for k, s in enumerate(self.segments)}

    # --- per-step ------------------------------------------------------------

    def tick(self) -> None:
        self.t += 1

    def _factor(self) -> np.ndarray:
        """(B, S) fade factor (1 for non-fade segments)."""
        steps = np.where(self.steps > 0, self.steps, 1)
        f = np.maximum(0.0, 1.0 - (self.t[:, None] - self.t0) / steps)
        return np.where(self.mode == FADE, f, 1.0)

    def disconnected(self) -> np.ndarray:
        """(B, S) bool: segment cut, noise or completely faded."""
        return (self.mode == CUT) | (self.mode == NOISE) | ((self.mode == FADE) & (self._factor() == 0.0))

    def fully_disconnected(self) -> np.ndarray:
        """(B,) bool: every sensory and motor segment disconnected."""
        return self.disconnected().all(axis=1)

    def sense(self, raw: np.ndarray, rows: Sequence[int] | None = None) -> np.ndarray:
        """Body -> mind for all environments (or only ``rows``; ``raw`` then has len(rows) rows).

        Updates embodied statistics from intact segments only.
        """
        rows = np.arange(self.n) if rows is None else np.asarray(rows, int)
        mask = self.intact_mask[rows]
        if mask.any():
            count, mean, m2 = self.count[rows], self.mean[rows], self.m2[rows]
            count[mask] += 1
            delta = raw[mask] - mean[mask]
            mean[mask] += delta / count[mask]
            m2[mask] += delta * (raw[mask] - mean[mask])
            self.count[rows], self.mean[rows], self.m2[rows] = count, mean, m2
        out = np.array(raw, dtype=float, copy=True)
        if mask.all():
            return out
        factor = self._factor()
        for i, r in enumerate(rows):
            for k in range(self.n_sensory):
                code = self.mode[r, k]
                if code == INTACT:
                    continue
                idx = self.segments[k].idx
                if code == CUT:
                    out[i, idx] = 0.0
                elif code == FADE:
                    out[i, idx] *= factor[r, k]
                else:
                    out[i, idx] = self.noise_mean[r, idx] + self.noise_std[r, idx] * \
                        self.rngs[r].standard_normal(len(idx))
        return out

    def act(self, intended: np.ndarray) -> np.ndarray:
        """Mind -> body for all environments. ``intended`` (B, 6) is left untouched."""
        applied = np.array(intended, dtype=float, copy=True)
        factor = self._factor()
        for k in range(self.n_sensory, len(self.segments)):
            idx = self.segments[k].idx
            m = self.mode[:, k]
            if not (m != INTACT).any():
                continue
            scale = np.where(m == CUT, 0.0, np.where(m == FADE, factor[:, k], 1.0))
            applied[:, idx] *= scale[:, None]
        return applied

    @staticmethod
    def void_action(intended: np.ndarray, applied: np.ndarray) -> dict[str, np.ndarray]:
        """Per part, the norm of what the mind attempted but did not reach the body; (B,) each."""
        out = {}
        for s in MOTOR_SEGMENTS:
            d = intended[..., s.idx] - applied[..., s.idx]
            out[s.part] = np.sqrt(np.sum(d * d, axis=-1))
        return out

    # --- state ---------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        return {
            "t": self.t.copy(), "mode": self.mode.copy(), "t0": self.t0.copy(), "steps": self.steps.copy(),
            "noise_mean": self.noise_mean.copy(), "noise_std": self.noise_std.copy(),
            "count": self.count.copy(), "mean": self.mean.copy(), "m2": self.m2.copy(),
            "intact_mask": self.intact_mask.copy(),
            "rngs": [json.dumps(r.bit_generator.state) for r in self.rngs],
        }

    def set_state(self, state: dict[str, Any]) -> None:
        for k in ("t", "mode", "t0", "steps", "noise_mean", "noise_std", "count", "mean", "m2", "intact_mask"):
            getattr(self, k)[...] = state[k]
        for r, s in zip(self.rngs, state["rngs"]):
            r.bit_generator.state = json.loads(s)


class _StatsView:
    """Embodied statistics of one environment (views into the batch arrays)."""

    def __init__(self, batch: BatchedBodyLink, row: int):
        self._b, self._r = batch, row

    @property
    def count(self) -> np.ndarray:
        return self._b.count[self._r]

    @property
    def mean(self) -> np.ndarray:
        return self._b.mean[self._r]

    @property
    def m2(self) -> np.ndarray:
        return self._b.m2[self._r]

    def std(self, idx: np.ndarray) -> np.ndarray:
        return self._b._std(np.array([self._r]), np.asarray(idx))[0]


class BodyLink:
    """One environment's link: a row of a ``BatchedBodyLink``.

    ``BodyLink(layout, rng)`` creates a standalone link (a batch of one).
    """

    def __init__(self, layout: ObsLayout, rng: np.random.Generator | None = None, *,
                 batch: BatchedBodyLink | None = None, row: int = 0):
        if batch is None:
            batch = BatchedBodyLink(layout, [rng if rng is not None else np.random.default_rng()])
        self._b, self._r = batch, row
        self.layout = layout
        self.stats = _StatsView(batch, row)

    @property
    def t(self) -> int:
        return int(self._b.t[self._r])

    @property
    def rng(self) -> np.random.Generator:
        return self._b.rngs[self._r]

    def set_mode(self, mode: str, channel: str | None = None, part: str | None = None,
                 steps: int | None = None) -> None:
        self._b.set_mode(mode, channel, part, steps, envs=[self._r])

    def modes(self) -> dict[str, str]:
        return self._b.modes(self._r)

    def fully_disconnected(self) -> bool:
        return bool(self._b.fully_disconnected()[self._r])

    # Per-step calls on a standalone link (inside an environment the batch is driven directly).
    def _standalone(self) -> None:
        if self._b.n != 1:
            raise RuntimeError("per-step calls are only for standalone links; the environment drives the batch")

    def tick(self) -> None:
        self._standalone()
        self._b.tick()

    def sense(self, raw: np.ndarray) -> np.ndarray:
        self._standalone()
        return self._b.sense(np.asarray(raw, float)[None])[0]

    def act(self, intended: np.ndarray) -> np.ndarray:
        self._standalone()
        return self._b.act(np.asarray(intended, float)[None])[0]

    @staticmethod
    def void_action(intended: np.ndarray, applied: np.ndarray) -> dict[str, float]:
        return {k: float(v) for k, v in BatchedBodyLink.void_action(intended, applied).items()}
