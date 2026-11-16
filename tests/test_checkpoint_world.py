"""Checkpoints v2: the agent with its world (env, policy, mind state)."""

import json

import numpy as np
import torch

from conftest import BASE_CONFIG, make_config, read_metrics
from umbral.checkpoint import clone, load_checkpoint, make_checkpoint, save_checkpoint
from umbral.cli import run_experiment
from umbral.config import load_config
from umbral.mind.model import build_model
from umbral.mind.policy import OUPolicy
from umbral.world.batched import BatchedUmbralEnv

TINY = ["train.n_envs=4", "train.seq_len=8", "train.episode_steps=16", "train.updates=6",
        "train.log_every=1", "train.checkpoint_every=3", "train.norm_envs=4", "train.norm_steps=40",
        "mind.encoder=16", "mind.hidden=16", "mind.head_hidden=16",
        "train.ckpt_eval_envs_per_history=1", "train.ckpt_eval_steps=30"]


def run_train(tmp_path, *overrides):
    cfg = load_config(BASE_CONFIG, [f"run.output_dir={tmp_path.as_posix()}", "experiment=train_world_model",
                                    *TINY, *overrides])
    return run_experiment(cfg)


def test_resume_mid_training_reproduces_the_continuation(tmp_path):
    full = run_train(tmp_path, "run.name=full")
    ckpt = full / "checkpoints" / "update_0000003.pt"
    assert ckpt.is_file()
    resumed = run_train(tmp_path, "run.name=resumed", f"run.resume_from={ckpt.as_posix()}")
    tail = [line for line in read_metrics(full) if json.loads(line)["step"] > 3]
    assert len(tail) > 0
    assert read_metrics(resumed) == tail
    # The final checkpoints agree too: world, policy and minds.
    a = load_checkpoint(full / "checkpoints" / "update_0000006.pt")
    b = load_checkpoint(resumed / "checkpoints" / "update_0000006.pt")
    for k in a.models:
        for name, t in a.models[k].items():
            assert torch.equal(t, b.models[k][name]), name
    for k, arr in a.env["arrays"].items():
        np.testing.assert_array_equal(arr, b.env["arrays"][k], err_msg=k)
    assert a.env["world_rngs"] == b.env["world_rngs"] and a.policy["rngs"] == b.policy["rngs"]


def agent_with_world(n=4, steps=40):
    cfg = make_config(*TINY)
    env = BatchedUmbralEnv(cfg, n, ["0", "1", "2", "3"][:n])
    env.reset(seed=[31, 32, 33, 34][:n])
    policy = OUPolicy(n, 0.9, 0.8, [41, 42, 43, 44][:n])
    model = build_model(env.layout, cfg.mind, efference_copy=True)
    h = model.initial_state(n)
    prev = np.zeros((n, 6))
    for _ in range(steps):
        obs, _, _, _, info = env.step(policy.act())
        h = step_mind(model, obs["flat"], prev, info["action_intended"], h)
        prev = info["action_intended"]
    return cfg, env, policy, model, h, prev


def step_mind(model, obs, prev, act, h):
    with torch.no_grad():
        out = model(torch.as_tensor(obs[:, None], dtype=torch.float32),
                    torch.as_tensor(prev[:, None], dtype=torch.float32),
                    torch.as_tensor(act[:, None], dtype=torch.float32), h)
    return out.h_last


def restore(cfg, ck, model_template):
    env = BatchedUmbralEnv(cfg, ck.env["n_envs"], ck.env["history_ids"])
    ck.restore_env(env)
    policy = OUPolicy(env.n, 0.9, 0.8, [0] * env.n)
    policy.set_state(ck.policy)
    model = build_model(env.layout, cfg.mind, efference_copy=True)
    ck.load_into({"mind": model})
    return env, policy, model, ck.mind_state["h"].clone(), ck.mind_state["prev_action"].numpy().copy()


def run_copy(cfg, ck, model_template, actions, mode_at=None):
    env, policy, model, h, prev = restore(cfg, ck, model_template)
    obs_log, h_log = [], []
    for t, a in enumerate(actions):
        if mode_at is not None and t == mode_at:
            env.link.set_mode("cut", channel="vision", envs=[0])
        obs, _, _, _, info = env.step(a)
        h = step_mind(model, obs["flat"], prev, info["action_intended"], h)
        prev = info["action_intended"]
        obs_log.append(obs["flat"].copy())
        h_log.append(h[0].numpy().copy())
    return np.array(obs_log), np.array(h_log)


def make_world_ckpt(tmp_path):
    cfg, env, policy, model, h, prev = agent_with_world()
    ck = make_checkpoint(40, {"mind": model}, {}, None, cfg, env=env, policy=policy,
                         mind_state={"h": h, "prev_action": torch.as_tensor(prev)})
    path = save_checkpoint(tmp_path / "agent.pt", ck)
    return cfg, path, model


def test_clone_with_world_gives_identical_trajectories(tmp_path):
    cfg, path, model = make_world_ckpt(tmp_path)
    actions = [np.random.default_rng(5).uniform(-1, 1, (4, 6)) for _ in range(60)]
    runs = [run_copy(cfg, c, model, actions) for c in clone(path, 3)]
    for obs, hs in runs[1:]:
        np.testing.assert_array_equal(obs, runs[0][0])
        np.testing.assert_array_equal(hs, runs[0][1])


def test_copies_diverge_only_from_the_step_the_mode_is_applied(tmp_path):
    cfg, path, model = make_world_ckpt(tmp_path)
    actions = [np.random.default_rng(6).uniform(-1, 1, (4, 6)) for _ in range(60)]
    k = 25
    a, b = clone(path, 2)
    obs_a, h_a = run_copy(cfg, a, model, actions)
    obs_b, h_b = run_copy(cfg, b, model, actions, mode_at=k)
    np.testing.assert_array_equal(obs_a[:k], obs_b[:k])
    np.testing.assert_array_equal(h_a[:k], h_b[:k])
    vision = slice(70, 105)
    assert (obs_b[k:, 0, vision] == 0).all() and not (obs_a[k:, 0, vision] == 0).all()
    assert not np.array_equal(h_a[k], h_b[k])
    np.testing.assert_array_equal(obs_a[:, 1:], obs_b[:, 1:])  # other environments unaffected


def test_v1_checkpoints_still_load(tmp_path):
    path = tmp_path / "v1.pt"
    torch.save({"format_version": 1, "step": 3, "models": {}, "optimizers": {}, "rng": {}, "config": {},
                "extra": {}}, path)
    ck = load_checkpoint(path)
    assert ck.step == 3 and ck.env is None and ck.policy is None and ck.mind_state is None
