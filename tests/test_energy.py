import numpy as np

from conftest import make_config, smooth_actions
from umbral.world.env import UmbralEnv


def test_energy_balance_adds_up():
    env = UmbralEnv(make_config("world.history_id=0"))
    env.reset(seed=2)
    p = env.params
    prev = env.body.energy
    eaten = 0
    for act in smooth_actions(3, 1500):
        _, _, _, _, info = env.step(act)
        b = info["energy_balance"]
        effort = float(np.mean(info["action_executed"] ** 2))
        assert b["before"] == prev
        assert np.isclose(b["basal"], p.basal)
        assert np.isclose(b["effort_cost"], p.effort_cost * effort)
        assert np.isclose(b["food_gain"], p.food_value * info["food_eaten"])
        assert np.isclose(b["unclipped"], b["before"] - b["basal"] - b["effort_cost"] + b["food_gain"])
        assert b["after"] == min(1.0, max(0.0, b["unclipped"])) == info["energy"]
        prev = info["energy"]
        eaten += info["food_eaten"]
    assert eaten > 0  # the food path was exercised


def test_energy_never_below_zero_and_body_still_moves():
    env = UmbralEnv(make_config("world.history_id=1", "body.energy.basal=0.02"))
    env.reset(seed=0)
    for act in smooth_actions(0, 200):
        _, _, _, _, info = env.step(act)
        assert info["energy"] >= 0.0
    assert env.body.energy == 0.0
    before = env.body.q.copy()
    _, _, _, _, info = env.step(np.array([0, 0, 1, 1, -1, -1]))
    assert info["energy"] == 0.0
    moved = np.abs(env.body.q - before).max()
    assert 0 < moved < env.params.max_joint_speed  # weaker, but never zero: no death


def test_eating_with_limb_tip_and_respawn():
    env = UmbralEnv(make_config("world.history_id=0", "body.energy.initial=0.5"))
    env.reset(seed=1)
    a = env.arena
    a.food_pos[0] = env._limbs[0, 2]  # a food item right at the left tip
    _, _, _, _, info = env.step(np.zeros(6))
    assert info["food_eaten"] == 1 and not a.food_active[0]
    assert np.isclose(info["energy_balance"]["food_gain"], env.params.food_value)
    respawn = int(env.history.cfg.food_respawn_steps)
    for _ in range(respawn - 1):
        env.step(np.zeros(6))
        assert not a.food_active[0]
    env.step(np.zeros(6))
    assert a.food_active[0]


def test_trunk_touch_does_not_eat():
    env = UmbralEnv(make_config("world.history_id=0"))
    env.reset(seed=1)
    env.arena.food_pos[0] = env.body.pos.copy()  # under the trunk, away from the tips
    _, _, _, _, info = env.step(np.zeros(6))
    assert info["food_eaten"] == 0
