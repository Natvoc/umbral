import itertools

import numpy as np

from conftest import make_config, smooth_actions
from umbral.world.env import UmbralEnv
from umbral.world.histories import build_arena

HISTORIES = ("0", "1", "2", "3")


def assert_states_equal(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], np.ndarray):
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)
        else:
            assert a[k] == b[k], k


def trajectory(hid, seed, n=400):
    env = UmbralEnv(make_config(f"world.history_id={hid}"))
    obs, _ = env.reset(seed=seed)
    flats, states = [obs["flat"]], []
    for act in smooth_actions(seed + 100, n):
        obs, _, _, _, _ = env.step(act)
        flats.append(obs["flat"])
        states.append(env.world_state())
    return np.array(flats), states


def test_same_seed_and_history_give_identical_trajectories():
    for hid in HISTORIES:
        f1, s1 = trajectory(hid, seed=5)
        f2, s2 = trajectory(hid, seed=5)
        np.testing.assert_array_equal(f1, f2)
        for a, b in zip(s1, s2):
            assert_states_equal(a, b)


def test_different_seeds_change_the_start_but_not_the_map():
    e1, e2 = (UmbralEnv(make_config("world.history_id=2")) for _ in range(2))
    e1.reset(seed=1)
    e2.reset(seed=2)
    np.testing.assert_array_equal(e1.arena.obstacle_pos, e2.arena.obstacle_pos)
    np.testing.assert_array_equal(e1.arena.food_pos, e2.arena.food_pos)
    assert not np.array_equal(e1.body.pos, e2.body.pos)


def test_four_distinct_histories():
    cfg = make_config()
    assert sorted(cfg.world.histories) == list(HISTORIES)
    arenas = {h: build_arena(make_config(f"world.history_id={h}").world)[0] for h in HISTORIES}
    names = {make_config(f"world.history_id={h}").world.histories[h].name for h in HISTORIES}
    assert len(names) == 4
    for a, b in itertools.combinations(HISTORIES, 2):
        A, B = arenas[a], arenas[b]
        same = (A.obstacle_pos.shape == B.obstacle_pos.shape and np.allclose(A.obstacle_pos, B.obstacle_pos)
                and A.food_pos.shape == B.food_pos.shape and np.allclose(A.food_pos, B.food_pos))
        assert not same, f"histories {a} and {b} build the same world"
    # And they differ in the intended way.
    assert len(arenas["0"].food_r) > len(arenas["1"].food_r)    # abundance vs scarcity
    assert len(arenas["2"].obstacle_r) > 15                     # corridor walls
    assert len(arenas["3"].neutral_r) >= 10                     # oasis: many neutral objects


def test_history_layout_is_reproducible():
    for h in HISTORIES:
        a1, _ = build_arena(make_config(f"world.history_id={h}").world)
        a2, _ = build_arena(make_config(f"world.history_id={h}").world)
        np.testing.assert_array_equal(a1.obstacle_pos, a2.obstacle_pos)
        np.testing.assert_array_equal(a1.neutral_pos, a2.neutral_pos)
        np.testing.assert_array_equal(a1.food_pos, a2.food_pos)


def test_reward_is_always_zero_and_no_termination():
    env = UmbralEnv(make_config())
    env.reset(seed=0)
    for act in smooth_actions(0, 200):
        _, reward, terminated, truncated, _ = env.step(act)
        assert reward == 0.0 and terminated is False and truncated is False
