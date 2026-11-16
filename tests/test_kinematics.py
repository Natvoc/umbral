import numpy as np

from conftest import empty_arena, make_config, smooth_actions
from umbral.body.body import BodyParams, BodyState, limb_points, move_body
from umbral.world.arena import unit
from umbral.world.env import UmbralEnv

P = BodyParams.from_config(make_config().body)


def test_straight_limbs_point_along_mount_angles():
    c = np.array([5.0, 5.0])
    pts = limb_points(P, c, 0.0, np.zeros((2, 2)))
    reach = P.trunk_radius + P.seg_lengths.sum()
    for i in range(2):
        np.testing.assert_allclose(pts[i, 0], c + P.trunk_radius * unit(P.mount_angles[i]))
        np.testing.assert_allclose(pts[i, 2], c + reach * unit(P.mount_angles[i]))


def test_tip_position_with_known_angles():
    c, h = np.array([3.0, 4.0]), 0.7
    q = np.array([[0.5, -0.9], [-1.1, 1.6]])
    pts = limb_points(P, c, h, q)
    for i in range(2):
        a0 = h + P.mount_angles[i]
        base = c + P.trunk_radius * np.array([np.cos(a0), np.sin(a0)])
        a1 = a0 + q[i, 0]
        elbow = base + P.seg_lengths[0] * np.array([np.cos(a1), np.sin(a1)])
        a2 = a1 + q[i, 1]
        tip = elbow + P.seg_lengths[1] * np.array([np.cos(a2), np.sin(a2)])
        np.testing.assert_allclose(pts[i], np.stack([base, elbow, tip]), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(pts[:, 1] - pts[:, 0], axis=1), P.seg_lengths[0])
    np.testing.assert_allclose(np.linalg.norm(pts[:, 2] - pts[:, 1], axis=1), P.seg_lengths[1])


def test_joint_limits_are_respected():
    arena = empty_arena()
    st = BodyState(np.array([5.0, 5.0]), 0.0, np.zeros((2, 2)), np.zeros((2, 2)), 1.0, 0.0)
    for sign in (1.0, -1.0):
        for _ in range(60):
            move_body(P, arena, st, np.array([0, 0, sign, sign, sign, sign]), 0.3)
            assert (st.q >= P.joint_lo - 1e-12).all() and (st.q <= P.joint_hi + 1e-12).all()
        expected = P.joint_hi if sign > 0 else P.joint_lo
        np.testing.assert_allclose(st.q, np.stack([expected, expected]))


def test_body_never_crosses_walls_or_obstacles():
    for hid in "0123":
        env = UmbralEnv(make_config(f"world.history_id={hid}"))
        env.reset(seed=4)
        a = env.arena
        R, lr = env.params.trunk_radius, env.params.limb_radius
        for act in smooth_actions(1, 1500, scale=1.0):
            env.step(act)
            pos = env.body.pos
            assert (pos >= R - 1e-9).all() and (pos <= a.size - R + 1e-9).all()
            gap = np.linalg.norm(a.obstacle_pos - pos, axis=1) - a.obstacle_r - R
            assert gap.min() > -1e-6, f"history {hid}: trunk inside an obstacle"
            pts = env._limbs[:, 1:].reshape(-1, 2)
            assert (pts >= lr - 1e-9).all() and (pts <= a.size - lr + 1e-9).all()
