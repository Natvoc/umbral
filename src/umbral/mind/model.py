"""The predictive mind (Phase 2a): a small recurrent forward model.

What enters where (the Phase 3 probes depend on this):

- **GRU input at step t**: ``[o_t, a_{t-1}]`` encoded by ``Linear -> ReLU``.
  ``o_t`` (107) is the observation as it arrived through the BodyLink;
  ``a_{t-1}`` (6) is the *intended* action of the previous step (efference copy
  of the last command; never the applied one). The hidden state ``h_t`` (256)
  therefore summarises everything up to the current observation.
- **Prediction head at step t**: ``[h_t, a_t]`` -> MLP -> prediction of
  ``o_{t+1}``. ``a_t`` is the intended action being sent now (efference copy of
  the current command). It enters the head, not the GRU, because in Phase 2b the
  action will be chosen *from* ``h_t``.
- Without efference copy (``efference_copy=False``, the control model) both
  ``a_{t-1}`` and ``a_t`` are replaced by zeros; everything else is identical.

``forward`` returns, besides the predictions, every hidden state ``h_t`` (B, T, H)
and the last one, for probing and for carrying state across segments.

Targets per channel (from ``o_t`` and ``o_{t+1}``):

- regression parts, predicted as the *change* ``o_{t+1} - o_t`` divided by a
  frozen per-element standard deviation ``delta_std`` (estimated once by a
  separate warm-up and stored in the model's state_dict, identical for compared
  models): proprio (10), intero (2), touch intensity (10), vision distance (7);
- categorical parts: touch type per sensor (6 classes: none, wall, food, obstacle,
  neutral, self) and vision type per ray (5 classes: none, wall, food, obstacle,
  neutral), cross-entropy. Also relative to the present: the logits are the head's
  output plus ``skip * one_hot(current type)``, with one learnable ``skip`` per
  channel (``touch_skip``, ``vision_skip``), so "nothing changes" is easy to
  express. (Without it the model lost to persistence on touch type in short runs.)

Loss = sum over the 4 channels (touch and vision add their regression and
classification terms). For comparisons, ``predict_next`` maps outputs back to
observation space (types as probabilities; bounded values clipped to their range)
and ``channel_mse`` gives the mean squared error per channel there - the same
metric for the model, the persistence baseline and the no-efference model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from umbral.body.sensors import N_TOUCH, TOUCH_WIDTH, VISION_WIDTH, ObsLayout

CHANNELS = ("proprio", "touch", "vision", "intero")


@dataclass(frozen=True)
class OutputLayout:
    """Indices tying model outputs to the observation layout."""
    layout: ObsLayout

    @property
    def n_rays(self) -> int:
        return self.layout.n_rays

    @property
    def reg_idx(self) -> np.ndarray:
        """Observation indices of the regression targets, in output order."""
        s = self.layout.slices
        t0, v0 = s["touch"].start, s["vision"].start
        return np.concatenate([
            np.arange(s["proprio"].start, s["proprio"].stop),            # 10
            np.arange(s["intero"].start, s["intero"].stop),              # 2
            t0 + TOUCH_WIDTH * np.arange(N_TOUCH),                       # 10 touch intensities
            v0 + VISION_WIDTH * np.arange(self.n_rays),                  # 7 vision distances
        ])

    @property
    def reg_channel(self) -> list[str]:
        return ["proprio"] * 10 + ["intero"] * 2 + ["touch"] * N_TOUCH + ["vision"] * self.n_rays

    @property
    def n_reg(self) -> int:
        return len(self.reg_idx)

    @property
    def n_out(self) -> int:
        return self.n_reg + N_TOUCH * TOUCH_WIDTH + self.n_rays * VISION_WIDTH


@dataclass
class WorldModelOutput:
    reg: torch.Tensor            # (B, T, n_reg) normalized predicted change
    touch_logits: torch.Tensor   # (B, T, 10, 6)
    vision_logits: torch.Tensor  # (B, T, R, 5)
    hidden: torch.Tensor         # (B, T, H) every h_t
    h_last: torch.Tensor         # (1, B, H)


class WorldModel(nn.Module):
    def __init__(self, layout: ObsLayout, action_dim: int = 6, encoder: int = 128, hidden: int = 256,
                 head_hidden: int = 128, efference_copy: bool = True):
        super().__init__()
        self.layout = layout
        self.out_layout = OutputLayout(layout)
        self.obs_dim, self.action_dim, self.hidden_size = layout.dim, action_dim, hidden
        self.efference_copy = bool(efference_copy)
        self.encoder = nn.Linear(self.obs_dim + action_dim, encoder)
        self.gru = nn.GRU(encoder, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + action_dim, head_hidden), nn.ReLU(),
                                  nn.Linear(head_hidden, self.out_layout.n_out))
        # Categorical parts are also predicted relative to the present: the current type
        # (one-hot, incl. "none") is added to the logits with a learnable weight per channel.
        self.touch_skip = nn.Parameter(torch.tensor(1.0))
        self.vision_skip = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("delta_std", torch.ones(self.out_layout.n_reg))
        self.register_buffer("reg_idx", torch.as_tensor(self.out_layout.reg_idx, dtype=torch.long))

    def set_delta_std(self, std: np.ndarray | torch.Tensor) -> None:
        self.delta_std.copy_(torch.as_tensor(std, dtype=self.delta_std.dtype))

    def initial_state(self, batch: int) -> torch.Tensor:
        return torch.zeros(1, batch, self.hidden_size)

    def forward(self, obs: torch.Tensor, prev_action: torch.Tensor, action: torch.Tensor,
                h0: torch.Tensor | None = None) -> WorldModelOutput:
        """obs (B,T,107), prev_action (B,T,6) = a_{t-1}, action (B,T,6) = a_t, h0 (1,B,H)."""
        if not self.efference_copy:
            prev_action = torch.zeros_like(prev_action)
            action = torch.zeros_like(action)
        x = F.relu(self.encoder(torch.cat([obs, prev_action], dim=-1)))
        hidden, h_last = self.gru(x, h0)
        y = self.head(torch.cat([hidden, action], dim=-1))
        B, T = obs.shape[:2]
        n_reg, R = self.out_layout.n_reg, self.out_layout.n_rays
        n_touch = N_TOUCH * TOUCH_WIDTH
        t_cls, v_cls = self._types(obs)
        touch_logits = y[..., n_reg:n_reg + n_touch].reshape(B, T, N_TOUCH, TOUCH_WIDTH) \
            + self.touch_skip * F.one_hot(t_cls, TOUCH_WIDTH).to(y.dtype)
        vision_logits = y[..., n_reg + n_touch:].reshape(B, T, R, VISION_WIDTH) \
            + self.vision_skip * F.one_hot(v_cls, VISION_WIDTH).to(y.dtype)
        return WorldModelOutput(reg=y[..., :n_reg], touch_logits=touch_logits, vision_logits=vision_logits,
                                hidden=hidden, h_last=h_last)

    # --- targets, losses, predictions ------------------------------------------------

    def _types(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Class targets: touch (B,T,10) in 0..5 and vision (B,T,R) in 0..4 (0 = none)."""
        s = self.layout.slices
        lead = obs.shape[:-1]
        touch = obs[..., s["touch"]].reshape(*lead, N_TOUCH, TOUCH_WIDTH)
        vision = obs[..., s["vision"]].reshape(*lead, self.out_layout.n_rays, VISION_WIDTH)
        t_cls = torch.where(touch[..., 0] > 0, 1 + touch[..., 1:].argmax(-1), torch.zeros_like(touch[..., 0],
                                                                                               dtype=torch.long))
        v_cls = torch.where(vision[..., 1:].sum(-1) > 0, 1 + vision[..., 1:].argmax(-1),
                            torch.zeros_like(vision[..., 0], dtype=torch.long))
        return t_cls, v_cls

    def loss(self, obs: torch.Tensor, next_obs: torch.Tensor, out: WorldModelOutput) -> dict[str, torch.Tensor]:
        """Per-channel losses and ``total`` (their sum)."""
        target = (next_obs[..., self.reg_idx] - obs[..., self.reg_idx]) / self.delta_std
        sq = (out.reg - target) ** 2
        n_p, n_i, n_t = 10, 2, N_TOUCH
        t_cls, v_cls = self._types(next_obs)
        losses = {
            "proprio": sq[..., :n_p].mean(),
            "intero": sq[..., n_p:n_p + n_i].mean(),
            "touch": sq[..., n_p + n_i:n_p + n_i + n_t].mean()
            + F.cross_entropy(out.touch_logits.reshape(-1, TOUCH_WIDTH), t_cls.reshape(-1)),
            "vision": sq[..., n_p + n_i + n_t:].mean()
            + F.cross_entropy(out.vision_logits.reshape(-1, VISION_WIDTH), v_cls.reshape(-1)),
        }
        losses["total"] = losses["proprio"] + losses["touch"] + losses["vision"] + losses["intero"]
        return losses

    def predict_next(self, obs: torch.Tensor, out: WorldModelOutput) -> torch.Tensor:
        """Predicted ``o_{t+1}`` in observation space (B, T, 107)."""
        pred = obs.clone()
        pred[..., self.reg_idx] = obs[..., self.reg_idx] + out.reg * self.delta_std
        s = self.layout.slices
        lead = obs.shape[:-1]
        touch = pred[..., s["touch"]].reshape(*lead, N_TOUCH, TOUCH_WIDTH).clone()
        touch[..., 1:] = F.softmax(out.touch_logits, dim=-1)[..., 1:]
        touch[..., 0] = touch[..., 0].clamp(0.0, 1.0)
        vision = pred[..., s["vision"]].reshape(*lead, self.out_layout.n_rays, VISION_WIDTH).clone()
        vision[..., 1:] = F.softmax(out.vision_logits, dim=-1)[..., 1:]
        vision[..., 0] = vision[..., 0].clamp(0.0, 1.0)
        pred[..., s["touch"]] = touch.reshape(*lead, -1)
        pred[..., s["vision"]] = vision.reshape(*lead, -1)
        pred[..., s["proprio"]] = pred[..., s["proprio"]].clamp(-1.0, 1.0)
        pred[..., s["intero"]] = pred[..., s["intero"]].clamp(0.0, 1.0)
        return pred


def channel_sq_error(layout: ObsLayout, pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray
                     ) -> dict[str, tuple[float, int]]:
    """Per channel: (sum of squared errors, number of elements), for exact aggregation."""
    pred_t = torch.as_tensor(pred, dtype=torch.float64)
    targ_t = torch.as_tensor(target, dtype=torch.float64)
    out = {}
    for ch in CHANNELS:
        d = pred_t[..., layout.slices[ch]] - targ_t[..., layout.slices[ch]]
        out[ch] = (float((d * d).sum()), d.numel())
    return out


def build_model(layout: ObsLayout, mind_cfg, efference_copy: bool) -> WorldModel:
    return WorldModel(layout, encoder=int(mind_cfg.encoder), hidden=int(mind_cfg.hidden),
                      head_hidden=int(mind_cfg.head_hidden), efference_copy=efference_copy)


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
