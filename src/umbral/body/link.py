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
``act`` and ``sense``. Noise draws use the link's own generator, so they never
change the world's random stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from umbral.body.sensors import CHANNELS, MOTOR_SEGMENTS, PARTS, ObsLayout, Segment

MODES = ("intact", "cut", "noise", "fade")
MOTOR = "motor"


@dataclass
class _SegState:
    seg: Segment
    mode: str = "intact"
    t0: int = 0
    steps: int = 0
    noise_mean: np.ndarray | None = None
    noise_std: np.ndarray | None = None


@dataclass
class _Stats:
    """Element-wise Welford accumulator over the embodied (intact) signal."""
    count: np.ndarray
    mean: np.ndarray
    m2: np.ndarray = field(default=None)  # type: ignore[assignment]

    @classmethod
    def zeros(cls, dim: int) -> "_Stats":
        return cls(np.zeros(dim), np.zeros(dim), np.zeros(dim))

    def update(self, x: np.ndarray, mask: np.ndarray) -> None:
        if not mask.any():
            return
        self.count[mask] += 1
        delta = x[mask] - self.mean[mask]
        self.mean[mask] += delta / self.count[mask]
        self.m2[mask] += delta * (x[mask] - self.mean[mask])

    def std(self, idx: np.ndarray) -> np.ndarray:
        return np.sqrt(self.m2[idx] / np.maximum(self.count[idx], 1))


class BodyLink:
    def __init__(self, layout: ObsLayout, rng: np.random.Generator):
        self.layout = layout
        self.rng = rng
        self.t = 0
        self._sensory = [_SegState(s) for s in layout.segments()]
        self._motor = [_SegState(s) for s in MOTOR_SEGMENTS]
        self.stats = _Stats.zeros(layout.dim)
        self._intact_mask = np.ones(layout.dim, bool)

    # --- configuration -------------------------------------------------------

    def _select(self, channel: str | None, part: str | None) -> list[_SegState]:
        if channel is not None and channel not in (*CHANNELS, MOTOR):
            raise ValueError(f"unknown channel {channel!r}")
        if part is not None and part not in PARTS:
            raise ValueError(f"unknown part {part!r}")
        segs = [s for s in (*self._sensory, *self._motor)
                if (channel is None or s.seg.channel == channel) and (part is None or s.seg.part == part)]
        if not segs:
            raise ValueError(f"no segment for channel={channel!r}, part={part!r}")
        return segs

    def set_mode(self, mode: str, channel: str | None = None, part: str | None = None,
                 steps: int | None = None) -> None:
        """Set ``mode`` on every segment matching ``channel`` and ``part`` (None = all)."""
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        if mode == "fade" and (steps is None or steps < 1):
            raise ValueError("fade needs steps >= 1")
        segs = self._select(channel, part)
        if mode == "noise":
            if any(s.seg.channel == MOTOR for s in segs):
                raise ValueError("noise is defined only for sensory channels")
            for s in segs:
                if (self.stats.count[s.seg.idx] < 2).any():
                    raise ValueError(f"no embodied statistics yet for {s.seg.channel}/{s.seg.part}")
        for s in segs:
            s.mode, s.t0, s.steps = mode, self.t, int(steps or 0)
            if mode == "noise":  # freeze embodied statistics now
                s.noise_mean = self.stats.mean[s.seg.idx].copy()
                s.noise_std = self.stats.std(s.seg.idx)
            else:
                s.noise_mean = s.noise_std = None
        self._intact_mask = np.zeros(self.layout.dim, bool)
        for s in self._sensory:
            if s.mode == "intact":
                self._intact_mask[s.seg.idx] = True

    def modes(self) -> dict[str, str]:
        return {f"{s.seg.channel}/{s.seg.part}": s.mode for s in (*self._sensory, *self._motor)}

    # --- per-step ------------------------------------------------------------

    def tick(self) -> None:
        self.t += 1

    def _factor(self, s: _SegState) -> float:
        return max(0.0, 1.0 - (self.t - s.t0) / s.steps)

    def _disconnected(self, s: _SegState) -> bool:
        return s.mode in ("cut", "noise") or (s.mode == "fade" and self._factor(s) == 0.0)

    def fully_disconnected(self) -> bool:
        """True when every sensory and motor segment is cut, noise or completely faded."""
        return all(self._disconnected(s) for s in (*self._sensory, *self._motor))

    def sense(self, raw: np.ndarray) -> np.ndarray:
        """Body -> mind. Updates embodied statistics from intact segments only."""
        self.stats.update(raw, self._intact_mask)
        out = np.array(raw, dtype=float, copy=True)
        for s in self._sensory:
            if s.mode == "intact":
                continue
            idx = s.seg.idx
            if s.mode == "cut":
                out[idx] = 0.0
            elif s.mode == "fade":
                out[idx] *= self._factor(s)
            else:  # noise
                out[idx] = s.noise_mean + s.noise_std * self.rng.standard_normal(len(idx))
        return out

    def act(self, intended: np.ndarray) -> np.ndarray:
        """Mind -> body. Returns the applied action; ``intended`` is left untouched."""
        applied = np.array(intended, dtype=float, copy=True)
        for s in self._motor:
            if s.mode == "cut":
                applied[s.seg.idx] = 0.0
            elif s.mode == "fade":
                applied[s.seg.idx] *= self._factor(s)
        return applied

    @staticmethod
    def void_action(intended: np.ndarray, applied: np.ndarray) -> dict[str, float]:
        """Per part, the norm of what the mind attempted but did not reach the body."""
        return {s.part: float(np.linalg.norm(intended[s.idx] - applied[s.idx])) for s in MOTOR_SEGMENTS}
