"""World freeze during total disembodiment (world.freeze_when_disembodied)."""

import numpy as np

from conftest import make_config, smooth_actions
from test_world_repro import assert_states_equal
from umbral.world.env import UmbralEnv


def env_after(steps, *overrides, seed=7):
    env = UmbralEnv(make_config("world.history_id=0", *overrides))
    env.reset(seed=seed)
    for act in smooth_actions(seed, steps):
        env.step(act)
    return env


def test_total_disconnection_freezes_world_and_restores_exactly():
    K, N, M = 200, 150, 300
    after = smooth_actions(11, M)

    ref = env_after(K)  # never disconnected
    ref_states = []
    for act in after:
        ref.step(act)
        ref_states.append(ref.world_state())

    env = env_after(K)
    env.link.set_mode("cut")  # every channel and every part, motor included
    frozen_state = env.world_state()
    for act in smooth_actions(99, N):
        obs, _, _, _, info = env.step(act)
        assert info["frozen"]
        assert (obs["flat"] == 0).all()
        np.testing.assert_array_equal(info["action_intended"], np.clip(act, -1, 1))
        assert sum(info["void_action"].values()) > 0
        assert_states_equal(env.world_state(), frozen_state)
    assert env.steps == K + N and env.world_steps == K

    env.link.set_mode("intact")
    for act, expected in zip(after, ref_states):
        _, _, _, _, info = env.step(act)
        assert not info["frozen"]
        assert_states_equal(env.world_state(), expected)


def test_noise_and_finished_fade_also_count_as_disembodied():
    env = env_after(300)
    env.link.set_mode("noise", channel="proprio")
    env.link.set_mode("noise", channel="touch")
    env.link.set_mode("fade", channel="vision", steps=5)
    env.link.set_mode("fade", channel="intero", steps=5)
    env.link.set_mode("cut", channel="motor")
    frozen = [env.step(np.zeros(6))[4]["frozen"] for _ in range(8)]
    assert frozen == [False] * 4 + [True] * 4  # frozen once the fades reach zero


def test_partial_loss_keeps_the_world_running():
    env = env_after(100)
    env.link.set_mode("cut", part="left_limb")
    e0, w0 = env.body.energy, env.world_steps
    for act in smooth_actions(1, 50):
        _, _, _, _, info = env.step(act)
        assert not info["frozen"]
    assert env.world_steps == w0 + 50 and env.body.energy < e0


def test_freeze_can_be_disabled():
    env = env_after(100, "world.freeze_when_disembodied=false")
    env.link.set_mode("cut")
    e0 = env.body.energy
    for _ in range(20):
        assert not env.step(np.zeros(6))[4]["frozen"]
    assert env.body.energy < e0  # metabolism keeps running
