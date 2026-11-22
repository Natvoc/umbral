"""A blocked trunk still turns in place (22 Nov 2026, docs/protocolo.md).

Before, a reverted trunk move also reverted the turn: in the pilot, 4 battery bodies
stayed wedged for 58 to 1,682 steps repeating the same blocked action while their
energy ran out. ``data/wedged_states_20261122.npz`` holds those 4 exact states
(captured with the old physics, commit e7bde1f) and the action each body repeated.
"""

from pathlib import Path

import numpy as np

from conftest import empty_arena, make_config, smooth_actions
from umbral.body.body import BodyParams, BodyState, _limbs_inside_b, move_body, move_body_b
from umbral.world.env import UmbralEnv

P = BodyParams.from_config(make_config().body)
WEDGED = Path(__file__).parent / "data" / "wedged_states_20261122.npz"
PUSH = float(make_config().world.neutral_push_fraction)


def test_a_body_blocked_at_the_wall_turns_in_place():
    # Facing the wall, left limb straight ahead-left with its tip at the wall: any step forward
    # would put it outside, so the translation is reverted. The turn alone is allowed.
    q = np.array([[0.0, 0.0], [-1.3, -2.2]])
    x = 9.03
    assert _limbs_inside_b(P, 10.0, np.array([[x, 5.0]]), np.zeros(1), q[None]).all()
    st = BodyState(np.array([x, 5.0]), 0.0, q.copy(), np.zeros((2, 2)), 1.0, 0.0)
    move_body(P, empty_arena(), st, np.array([1.0, 0.0, 0, 0, 0, 0]), PUSH)
    assert np.allclose(st.pos, [x, 5.0]) and st.heading == 0.0  # forward only: blocked
    move_body(P, empty_arena(), st, np.array([1.0, 0.5, 0, 0, 0, 0]), PUSH)
    assert np.allclose(st.pos, [x, 5.0])  # it does not advance ...
    assert np.isclose(st.heading, 0.5 * P.max_turn)  # ... but turns in place (before: nothing at all)
    for _ in range(5):  # repeating the same action, it turns until it can move on
        move_body(P, empty_arena(), st, np.array([1.0, 0.5, 0, 0, 0, 0]), PUSH)
    assert np.linalg.norm(st.pos - [x, 5.0]) > 0.1
    st = BodyState(np.array([x, 5.0]), 0.0, q.copy(), np.zeros((2, 2)), 1.0, 0.0)
    move_body(P, empty_arena(), st, np.array([1.0, -0.5, 0, 0, 0, 0]), PUSH)
    assert np.allclose(st.pos, [x, 5.0]) and st.heading == 0.0  # turning the limb into the wall: reverted too


def test_a_turn_that_would_put_a_limb_outside_is_reverted_too():
    # Search near the walls for blocked moves whose turn is not allowed either: the body keeps
    # its heading and every limb point stays inside (the old rule for those cases).
    rng = np.random.default_rng(0)
    R, size = P.trunk_radius, 10.0
    found = 0
    for _ in range(4000):
        pos = np.array([[size - R - rng.uniform(0, 0.05), rng.uniform(1, 9)]])
        heading, q = rng.uniform(-np.pi, np.pi, 1), rng.uniform(P.joint_lo, P.joint_hi, (1, 2, 2))
        if not _limbs_inside_b(P, size, pos, heading, q).all():
            continue
        a = np.array([[1.0, rng.choice([-1.0, 1.0]), 0, 0, 0, 0]])
        z2 = np.zeros((1, 0, 2))
        out = move_body_b(P, size, pos, heading, q, np.ones(1), np.zeros(1), z2, np.zeros((1, 0)), z2,
                          np.zeros((1, 0)), a, PUSH)
        assert _limbs_inside_b(P, size, out["pos"], out["heading"], out["q"]).all()
        if np.allclose(out["pos"], pos) and np.isclose(out["heading"][0], heading[0]):
            found += 1
    assert found > 0  # the case exists and is handled


def _wedged():
    d = np.load(WEDGED)
    for name in d["names"]:
        yield str(name), {k.split("__")[1]: d[k] for k in d.files if k.startswith(f"{name}__")}


def _step(s: dict, a: np.ndarray) -> dict:
    out = move_body_b(P, 10.0, s["pos"][None], np.atleast_1d(s["heading"]), s["q"][None], np.atleast_1d(s["energy"]),
                      np.atleast_1d(s["fatigue"]), s["obstacle_pos"][None], s["obstacle_r"][None],
                      s["neutral_pos"][None], s["neutral_r"][None], a[None], PUSH)
    return {**s, "pos": out["pos"][0], "heading": out["heading"][0], "q": out["q"][0],
            "neutral_pos": out["neutral_pos"][0]}


def _gets_out(s: dict, a: np.ndarray, steps: int = 200) -> bool:
    cur = s
    for _ in range(steps):
        cur = _step(cur, a)
        if np.linalg.norm(cur["pos"] - s["pos"]) > 0.05:
            return True
    return False


def test_the_four_wedged_bodies_of_the_pilot_now_get_out_by_turning():
    """Every wedged state has a way out by turning (its own forward command, the turn reversed).

    With the very action the body repeated, only the one wedged against obstacles gets out:
    in the other 3 a limb lies against the wall and the turn the policy asks for would put
    it outside the arena, so that turn is reverted too (the rule as decided).
    """
    own_action_frees = {}
    for name, s in _wedged():
        a = s["applied"]
        away = a.copy()
        away[1] = -a[1]
        nxt = _step(s, away)
        assert not np.isclose(nxt["heading"], s["heading"]), f"{name}: it can turn in place"
        assert _gets_out(s, away), f"{name}: turning, it gets out"
        own_action_frees[name] = _gets_out(s, a)
    assert own_action_frees == {"curioso_corta": True, "curioso_larga": False, "cauteloso_corta": False,
                                "equilibrado_larga": False}


def test_turning_in_place_never_crosses_walls_or_obstacles():
    for hid in "0123":
        env = UmbralEnv(make_config(f"world.history_id={hid}"))
        env.reset(seed=11)
        a = env.arena
        R, lr = env.params.trunk_radius, env.params.limb_radius
        for act in smooth_actions(3, 3000, scale=1.5):  # strong actions: many blocked moves
            env.step(act)
            pos = env.body.pos
            assert (pos >= R - 1e-9).all() and (pos <= a.size - R + 1e-9).all()
            gap = np.linalg.norm(a.obstacle_pos - pos, axis=1) - a.obstacle_r - R
            assert gap.min() > -1e-6
            pts = env._limbs[:, 1:].reshape(-1, 2)
            assert (pts >= lr - 1e-9).all() and (pts <= a.size - lr + 1e-9).all()
