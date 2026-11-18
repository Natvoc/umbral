"""The mind that decides (Phase 2b): the predictive model plus an actor and a critic.

    o_t, a_{t-1} --encoder, GRU--> h_t --prediction head [h_t, a_t]--> o_{t+1}
                                     |
                                     +--(stop gradient)--> actor  -> a_t
                                     +--(stop gradient)--> critic -> V(h_t)

**Separation of the self-model from reward.** The encoder, the GRU and the
prediction head (``self.world``, the Phase 2a ``WorldModel`` with efference copy)
are trained *only* by the prediction loss. The actor and the critic read ``h_t``
detached: no gradient of the PPO loss reaches the self-model. The bodily self is
therefore what the agent learns by predicting its body. The character lives in
how it decides (the actor, and the drives' weights that shaped it). This
separation is what later allows measuring, separately, which of the two falls
first when the body is lost.

The action ``a_t`` is drawn from the actor at ``h_t`` and enters the prediction
head as the efference copy, exactly as designed in Phase 2a. It is a tanh-squashed
Gaussian: ``u ~ N(mu(h_t), sigma)``, ``a_t = tanh(u)``, with the change-of-variables
correction ``log pi(a) = log N(u) - sum log(1 - tanh(u)**2)``. The noise comes from
one numpy Generator per environment (``GaussianNoise``), not from torch.
"""

from __future__ import annotations

import json
import math
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from umbral.mind.model import WorldModel


def _mlp(inp: int, hidden: int, out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inp, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, out))


class ActorCritic(nn.Module):
    def __init__(self, hidden_size: int, action_dim: int = 6, width: int = 128, init_log_std: float = -0.5):
        super().__init__()
        self.actor = _mlp(hidden_size, width, action_dim)
        self.critic = _mlp(hidden_size, width, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))
        with torch.no_grad():  # small initial means: the first actions are mostly the noise
            self.actor[-1].weight.mul_(0.01)
            self.actor[-1].bias.zero_()

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean of ``u`` and value, from ``h`` *detached* (the self-model gets no gradient)."""
        h = h.detach()
        return self.actor(h), self.critic(h).squeeze(-1)

    def log_prob(self, mu: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """log pi(tanh(u)) for pre-squash samples ``u``; sums over the action dimensions."""
        std = self.log_std.exp()
        lp = -0.5 * ((u - mu) / std) ** 2 - self.log_std - 0.5 * math.log(2 * math.pi)
        # log(1 - tanh(u)^2), numerically stable form
        corr = 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
        return (lp - corr).sum(-1)

    def entropy(self) -> torch.Tensor:
        """Entropy of the pre-squash Gaussian (the usual PPO bonus surrogate)."""
        return (0.5 + 0.5 * math.log(2 * math.pi) + self.log_std).sum()


class GaussianNoise:
    """Standard normal noise for the actor, one Generator per environment (checkpointed)."""

    def __init__(self, n_envs: int, seeds: Sequence[int], action_dim: int = 6):
        if len(seeds) != n_envs:
            raise ValueError("one seed per environment")
        self.n, self.action_dim = int(n_envs), int(action_dim)
        self.rngs = [np.random.default_rng(np.random.SeedSequence(int(s))) for s in seeds]

    def sample(self) -> np.ndarray:
        return np.stack([g.standard_normal(self.action_dim) for g in self.rngs])

    def get_state(self) -> dict[str, Any]:
        return {"rngs": [json.dumps(g.bit_generator.state) for g in self.rngs]}

    def set_state(self, state: dict[str, Any]) -> None:
        for g, s in zip(self.rngs, state["rngs"]):
            g.bit_generator.state = json.loads(s)


class Agent(nn.Module):
    """``world`` (self-model, prediction loss only) + ``policy`` (actor-critic, PPO loss only)."""

    def __init__(self, world: WorldModel, width: int = 128, init_log_std: float = -0.5):
        super().__init__()
        if not world.efference_copy:
            raise ValueError("the agent's world model must use the efference copy")
        self.world = world
        self.policy = ActorCritic(world.hidden_size, world.action_dim, width, init_log_std)

    @torch.no_grad()
    def act(self, h: torch.Tensor, eps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """From ``h_t`` (B,H) and noise (B,6): action ``a_t``, pre-squash ``u``, log-prob, value."""
        mu, v = self.policy(h)
        u = mu + self.policy.log_std.exp() * torch.as_tensor(eps, dtype=torch.float32)
        logp = self.policy.log_prob(mu, u)
        return torch.tanh(u).double().numpy(), u.numpy(), logp.numpy(), v.numpy()


def build_agent(world: WorldModel, agent_cfg) -> Agent:
    return Agent(world, width=int(agent_cfg.width), init_log_std=float(agent_cfg.init_log_std))
