import numpy as np
import pytest

from conftest import make_config, smooth_actions
from umbral.body.link import MODES, BodyLink
from umbral.body.sensors import CHANNELS, PARTS, ObsLayout
from umbral.world.env import UmbralEnv

LAYOUT = ObsLayout(7)


def seg_idx(channel, part):
    return next(s.idx for s in LAYOUT.segments() if s.channel == channel and s.part == part)


def lived_env(steps=600, seed=0):
    """An environment after some embodied life (link statistics filled)."""
    env = UmbralEnv(make_config())
    env.reset(seed=seed)
    for act in smooth_actions(seed, steps):
        env.step(act)
    return env


def test_segments_cover_the_observation_exactly_once():
    idx = np.concatenate([s.idx for s in LAYOUT.segments()])
    assert sorted(idx.tolist()) == list(range(LAYOUT.dim))
    assert LAYOUT.dim == 107


def test_intact_changes_nothing():
    link = BodyLink(LAYOUT, np.random.default_rng(0))
    raw = np.random.default_rng(1).normal(size=LAYOUT.dim)
    np.testing.assert_array_equal(link.sense(raw), raw)
    action = np.array([0.3, -0.2, 0.9, -1.0, 0.1, 0.5])
    np.testing.assert_array_equal(link.act(action), action)


def test_cut_gives_zeros_only_on_its_segment():
    link = BodyLink(LAYOUT, np.random.default_rng(0))
    raw = np.random.default_rng(1).normal(size=LAYOUT.dim) + 3.0
    link.set_mode("cut", channel="touch", part="left_limb")
    out = link.sense(raw)
    idx = seg_idx("touch", "left_limb")
    assert (out[idx] == 0).all()
    others = np.setdiff1d(np.arange(LAYOUT.dim), idx)
    np.testing.assert_array_equal(out[others], raw[others])


def test_cut_whole_channel_and_whole_part():
    link = BodyLink(LAYOUT, np.random.default_rng(0))
    raw = np.ones(LAYOUT.dim)
    link.set_mode("cut", channel="vision")
    assert (link.sense(raw)[LAYOUT.slices["vision"]] == 0).all()
    link.set_mode("intact")
    link.set_mode("cut", part="right_limb")
    out = link.sense(raw)
    for ch in ("proprio", "touch"):
        assert (out[seg_idx(ch, "right_limb")] == 0).all()
    assert (link.act(np.ones(6))[4:] == 0).all()


def test_noise_has_embodied_scale_and_frozen_statistics():
    env = lived_env()
    link = env.link
    idx = seg_idx("proprio", "left_limb")
    mean_before = link.stats.mean[idx].copy()
    std_before = link.stats.std(idx)
    count_before = link.stats.count[idx].copy()
    assert (std_before > 0.05).all()

    link.set_mode("noise", channel="proprio", part="left_limb")
    samples = []
    for act in smooth_actions(9, 3000):
        obs, _, _, _, _ = env.step(act)
        samples.append(obs["flat"][idx])
    samples = np.array(samples)
    np.testing.assert_allclose(samples.std(0), std_before, rtol=0.08)
    assert (np.abs(samples.mean(0) - mean_before) < 4 * std_before / np.sqrt(len(samples))).all()
    # Frozen: the signal computed while in noise mode never fed the statistics.
    np.testing.assert_array_equal(link.stats.count[idx], count_before)
    np.testing.assert_array_equal(link.stats.mean[idx], mean_before)
    # Unstructured: no temporal correlation.
    centred = samples - samples.mean(0)
    lag1 = (centred[1:] * centred[:-1]).mean(0) / centred.var(0)
    assert (np.abs(lag1) < 0.08).all()


def test_noise_rules():
    link = BodyLink(LAYOUT, np.random.default_rng(0))
    with pytest.raises(ValueError, match="no embodied statistics"):
        link.set_mode("noise", channel="vision")
    with pytest.raises(ValueError, match="sensory"):
        lived_env(50).link.set_mode("noise", channel="motor")


def test_noise_does_not_change_the_world_random_stream():
    e1, e2 = lived_env(300, seed=3), lived_env(300, seed=3)
    e2.link.set_mode("noise", channel="vision")
    for act in smooth_actions(4, 300):
        e1.step(act)
        e2.step(act)
    assert e1.world_state()["rng"] == e2.world_state()["rng"]
    np.testing.assert_array_equal(e1.body.pos, e2.body.pos)


def test_fade_decreases_monotonically_to_zero():
    link = BodyLink(LAYOUT, np.random.default_rng(0))
    N = 10
    link.set_mode("fade", channel="intero", steps=N)
    link.set_mode("fade", channel="motor", part="trunk", steps=N)
    raw = np.ones(LAYOUT.dim)
    sens, mot = [], []
    for _ in range(N + 5):
        link.tick()
        sens.append(link.sense(raw)[LAYOUT.slices["intero"]][0])
        mot.append(link.act(np.ones(6))[0])
    expected = np.maximum(0.0, 1.0 - np.arange(1, N + 6) / N)
    np.testing.assert_allclose(sens, expected)
    np.testing.assert_allclose(mot, expected)
    assert all(a > b for a, b in zip(sens[:N], sens[1:N]))
    assert sens[N - 1] == 0.0 and all(v == 0.0 for v in sens[N - 1:])


def test_disconnected_limb_does_not_move_and_intent_is_recorded():
    env = lived_env(100)
    env.link.set_mode("cut", channel="motor", part="left_limb")
    q_left = env.body.q[0].copy()
    q_right = env.body.q[1].copy()
    action = np.array([0.0, 0.0, 0.8, -0.6, 0.7, 0.7])
    _, _, _, _, info = env.step(action)
    np.testing.assert_array_equal(env.body.q[0], q_left)
    assert not np.array_equal(env.body.q[1], q_right)
    np.testing.assert_array_equal(info["action_intended"], action)
    np.testing.assert_array_equal(info["action_applied"][2:4], 0.0)
    assert np.isclose(info["void_action"]["left_limb"], np.linalg.norm([0.8, -0.6]))
    assert info["void_action"]["right_limb"] == 0.0 and info["void_action"]["trunk"] == 0.0


def test_observation_shape_is_stable_in_every_mode():
    env = lived_env(100)
    ref = {k: v.shape for k, v in env.layout.split(np.zeros(env.observation_dim)).items()}
    assert ref["flat"] == (107,)
    for mode in MODES:
        for channel in CHANNELS:
            for part in (None, *PARTS):
                try:
                    env.link.set_mode(mode, channel=channel, part=part, steps=5)
                except ValueError:
                    continue  # e.g. vision has no limb segment
                obs, _, _, _, _ = env.step(np.zeros(6))
                assert {k: v.shape for k, v in obs.items()} == ref
                assert obs["flat"].dtype == np.float64 and np.isfinite(obs["flat"]).all()
                env.link.set_mode("intact")
