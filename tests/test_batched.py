"""BatchedUmbralEnv: environment i of a batch == a single UmbralEnv with the same seed, history and actions."""

import numpy as np

from conftest import make_config, smooth_actions
from test_world_repro import assert_states_equal
from umbral.world.batched import BatchedUmbralEnv
from umbral.world.env import UmbralEnv

HISTORIES = ["0", "1", "2", "3", "2", "0", "3", "1"]
SEEDS = [11, 12, 13, 14, 15, 16, 17, 18]


def test_batch_rows_equal_single_envs_in_all_histories_with_freeze():
    cfg = make_config()
    n, steps = len(HISTORIES), 600
    actions = [smooth_actions(100 + i, steps, scale=1.0) for i in range(n)]
    cut_at, restore_at = 200, 320          # total disconnection window for some envs
    disconnected = {1, 2, 6}

    batch = BatchedUmbralEnv(cfg, n, HISTORIES)
    b_obs, _ = batch.reset(seed=SEEDS)
    singles = [UmbralEnv(cfg, history_id=h) for h in HISTORIES]
    s_obs = [env.reset(seed=s)[0] for env, s in zip(singles, SEEDS)]
    for i in range(n):
        np.testing.assert_array_equal(b_obs["flat"][i], s_obs[i]["flat"])

    for t in range(steps):
        if t == cut_at:
            batch.link.set_mode("cut", envs=sorted(disconnected))
            for i in disconnected:
                singles[i].link.set_mode("cut")
        if t == restore_at:
            batch.link.set_mode("intact", envs=sorted(disconnected))
            for i in disconnected:
                singles[i].link.set_mode("intact")
        b_obs, _, _, _, b_info = batch.step(np.stack([actions[i][t] for i in range(n)]))
        for i, env in enumerate(singles):
            obs, _, _, _, info = env.step(actions[i][t])
            np.testing.assert_array_equal(b_obs["flat"][i], obs["flat"], err_msg=f"env {i} step {t}")
            assert bool(b_info["frozen"][i]) == info["frozen"]
            assert int(b_info["food_eaten"][i]) == info["food_eaten"]
            assert_states_equal(batch.world_state(i), env.world_state())
    frozen_steps = restore_at - cut_at
    for i in range(n):
        expected = steps - (frozen_steps if i in disconnected else 0)
        assert int(batch.world_steps[i]) == expected


def test_partial_reset_matches_fresh_single_env():
    cfg = make_config()
    batch = BatchedUmbralEnv(cfg, 4, ["0", "1", "2", "3"])
    batch.reset(seed=[1, 2, 3, 4])
    for a in smooth_actions(0, 50):
        batch.step(np.tile(a, (4, 1)))
    obs, _ = batch.reset(seed=[99], envs=[2])
    single = UmbralEnv(cfg, history_id="2")
    s_obs, _ = single.reset(seed=99)
    np.testing.assert_array_equal(obs["flat"][2], s_obs["flat"])
    for a in smooth_actions(5, 100):
        b_obs, *_ = batch.step(np.tile(a, (4, 1)))
        obs1, *_ = single.step(a)
        np.testing.assert_array_equal(b_obs["flat"][2], obs1["flat"])


def test_probe_keeps_batch_equivalence_and_is_seen():
    """The behaviour-battery probe (an extra neutral slot) obeys the same equivalence."""
    cfg = make_config()
    batch = BatchedUmbralEnv(cfg, 4, ["0", "1", "2", "3"])
    batch.reset(seed=[5, 6, 7, 8])
    singles = [UmbralEnv(cfg, history_id=h) for h in ["0", "1", "2", "3"]]
    for env, s in zip(singles, [5, 6, 7, 8]):
        env.reset(seed=s)
    acts = [smooth_actions(30 + i, 120) for i in range(4)]
    seen = False
    for t in range(120):
        if t == 40:
            for i in (1, 3):
                ahead = batch.pos[i] + 1.5 * np.array([np.cos(batch.heading[i]), np.sin(batch.heading[i])])
                batch.place_probe(i, ahead, 0.3)
                singles[i].batch.place_probe(0, ahead.copy(), 0.3)
        if t == 90:
            batch.clear_probe(1)
            singles[1].batch.clear_probe(0)
        b_obs, *_ = batch.step(np.stack([acts[i][t] for i in range(4)]))
        for i, env in enumerate(singles):
            obs, *_ = env.step(acts[i][t])
            np.testing.assert_array_equal(b_obs["flat"][i], obs["flat"], err_msg=f"env {i} step {t}")
        if t == 40:
            seen = bool((b_obs["vision"][1, :, 4] == 1).any())  # the neutral column of some ray
    assert seen
    assert not batch.neutral_on[[0, 2], batch.probe_slot].any()
