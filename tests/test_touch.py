import numpy as np

from conftest import empty_arena, make_config
from umbral.body.body import BodyParams, BodyState, limb_points
from umbral.body.sensors import touch_sensors
from umbral.world.arena import NEUTRAL, OBSTACLE, SELF, WALL, unit

P = BodyParams.from_config(make_config().body)
R, LR, TR = P.trunk_radius, P.limb_radius, P.touch_range


def body_at(x, y, heading=0.0, q=None):
    q = np.zeros((2, 2)) if q is None else q
    st = BodyState(np.array([x, y], dtype=float), heading, q, np.zeros((2, 2)), 1.0, 0.0)
    return st, limb_points(P, st.pos, heading, q)


def run_touch(arena, st, limbs):
    return touch_sensors(P, arena, st, limbs, arena.objects())


def active(touch):
    """{sensor index: kind} for every active sensor."""
    return {int(k): int(np.argmax(touch[k, 1:])) for k in np.flatnonzero(touch[:, 0] > 0)}


def test_nothing_near_means_no_touch():
    st, limbs = body_at(5, 5)
    touch = run_touch(empty_arena(), st, limbs)
    assert active(touch) == {} and not touch.any()


def test_wall_behind_activates_only_back_sensor():
    st, limbs = body_at(R + 0.03, 5.0, heading=0.0)  # left wall just behind (relative 180 deg)
    touch = run_touch(empty_arena(), st, limbs)
    assert active(touch) == {4: WALL}
    assert np.isclose(touch[4, 0], 1 - 0.03 / TR)


def test_obstacle_on_left_and_neutral_ahead_with_rotated_heading():
    arena = empty_arena()
    h = 0.9
    st, limbs = body_at(5.0, 5.0, heading=h)
    limbs = limbs.copy()  # tips placed by hand, behind the body and away from everything
    limbs[0, 2] = st.pos + 1.25 * unit(h - 2.5)
    limbs[1, 2] = st.pos + 1.25 * unit(h + 2.5)
    arena.obstacle_pos = np.array([st.pos + (R + 0.4) * unit(h + np.pi / 2)])  # touching, sensor 2
    arena.obstacle_r = np.array([0.4])
    arena.neutral_pos = np.array([st.pos + (R + 0.3 + 0.04) * unit(h)])        # gap 0.04, sensor 0
    arena.neutral_r = np.array([0.3])
    touch = run_touch(arena, st, limbs)
    assert active(touch) == {2: OBSTACLE, 0: NEUTRAL}
    assert np.isclose(touch[2, 0], 1.0)
    assert np.isclose(touch[0, 0], 1 - 0.04 / TR)


def test_limb_tip_on_obstacle():
    arena = empty_arena()
    st, limbs = body_at(5.0, 5.0)
    tip = limbs[1, 2]  # right tip
    arena.obstacle_pos = np.array([tip + (0.3 + LR) * unit(-np.pi / 4)])
    arena.obstacle_r = np.array([0.3])
    assert active(run_touch(arena, st, limbs)) == {9: OBSTACLE}


def test_self_touch_is_felt_on_both_sides():
    st, limbs = body_at(5.0, 5.0)
    limbs = limbs.copy()
    limbs[0, 1] = st.pos + (R + 0.5) * unit(np.pi / 2)
    limbs[0, 2] = st.pos + (R + LR) * unit(3 * np.pi / 4)  # left tip resting on the trunk at 135 deg
    touch = run_touch(empty_arena(), st, limbs)
    assert active(touch) == {3: SELF, 8: SELF}
    assert np.isclose(touch[3, 0], 1.0) and np.isclose(touch[8, 0], 1.0)


def test_tips_touching_each_other():
    st, limbs = body_at(5.0, 5.0)
    limbs = limbs.copy()
    meet = st.pos + np.array([1.2, 0.0])
    limbs[0, 2] = meet + np.array([0.0, LR])
    limbs[1, 2] = meet - np.array([0.0, LR])
    assert active(run_touch(empty_arena(), st, limbs)) == {8: SELF, 9: SELF}
