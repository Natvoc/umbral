"""Phase 2b: the deciding mind, the drives, the agent's life and the population queue."""

import json

import numpy as np
import pytest
import torch

from conftest import BASE_CONFIG, make_config, read_metrics
from test_checkpoint_world import TINY
from umbral.body.sensors import ObsLayout
from umbral.checkpoint import load_checkpoint
from umbral.cli import run_experiment
from umbral.config import load_config
from umbral.experiments.agent import load_agent
from umbral.mind.agent import Agent
from umbral.mind.drives import (Drives, RewardScaler, Temperament, caution_reward, hunger_gate, hunger_reward,
                                world_prediction_error)
from umbral.mind.life import ppo_loss
from umbral.mind.model import build_model

LAYOUT = ObsLayout(7)
TINY_LIFE = ["agent.start=null", "mind.encoder=16", "mind.hidden=16", "mind.head_hidden=16", "agent.width=16",
             "life.n_envs=4", "life.rollout=16", "life.seq_len=8", "life.episode_steps=32", "life.iterations=6",
             "life.log_every=1", "life.checkpoint_every=2", "ppo.minibatches=2", "ppo.epochs=2",
             "behavior.episodes=2", "behavior.steps=60", "behavior.probe_offset=10", "behavior.probe_every=25",
             "behavior.probe_window=10", "behavior.n_boot=50", "eval.envs_per_history=1", "eval.steps=20",
             "eval.chunk=10", "eval.n_boot=50"]


@pytest.fixture(scope="module")
def infancy(tmp_path_factory):
    d = tmp_path_factory.mktemp("infancy")
    cfg = load_config(BASE_CONFIG, [f"run.output_dir={d.as_posix()}", "experiment=train_world_model", *TINY])
    run = run_experiment(cfg)
    return (run / "checkpoints" / "update_0000006.pt").as_posix()


def life_config(tmp_path, infancy, *overrides):
    return load_config(BASE_CONFIG, [f"run.output_dir={tmp_path.as_posix()}", "experiment=live_agent",
                                     f"agent.infancy={infancy}", *TINY_LIFE, *overrides])


# --- the mind ---------------------------------------------------------------------

def test_ppo_gradient_does_not_reach_the_self_model():
    cfg = make_config()
    torch.manual_seed(0)
    agent = Agent(build_model(LAYOUT, cfg.mind, True))
    obs = torch.rand(4, 5, 107)
    act = torch.rand(4, 5, 6) * 2 - 1
    out = agent.world(obs, torch.zeros_like(act), act)
    h = out.hidden.reshape(20, -1)  # still attached to the GRU's graph
    u = torch.randn(20, 6)
    loss, _ = ppo_loss(agent, h, u, torch.zeros(20), torch.randn(20), torch.randn(20), 0.2, 0.5, 0.01)
    loss.backward()
    for name, p in agent.world.named_parameters():
        assert p.grad is None or not p.grad.any(), name
    assert any(p.grad is not None and p.grad.any() for p in agent.policy.parameters())


def test_tanh_gaussian_log_prob_matches_the_change_of_variables():
    cfg = make_config()
    torch.manual_seed(1)
    agent = Agent(build_model(LAYOUT, cfg.mind, True))
    mu, u = torch.randn(3, 6), torch.randn(3, 6)
    std = agent.policy.log_std.exp()
    a = torch.tanh(u)
    expected = (torch.distributions.Normal(mu, std).log_prob(u) - torch.log(1 - a ** 2)).sum(-1)
    torch.testing.assert_close(agent.policy.log_prob(mu, u), expected, rtol=1e-4, atol=1e-4)


# --- the drives -----------------------------------------------------------------------

def test_hunger_is_drive_reduction():
    # D(0.5) = 0.09, D(0.6) = 0.04 with E* = 0.8 -> 10 * 0.05
    assert np.isclose(hunger_reward(np.array(0.5), np.array(0.6), 0.8, 10.0), 0.5)
    assert np.isclose(hunger_reward(np.array(0.9), np.array(0.95), 0.8, 1.0), 0.01 - 0.0225)
    assert hunger_reward(np.array(0.8), np.array(0.8), 0.8, 1.0) == 0.0


def test_caution_uses_the_intended_action():
    a = np.array([[1.0, 0, 0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
    np.testing.assert_allclose(caution_reward(a, 0.0016, 100.0), [-0.16 / 6, -0.16 * 0.25])


def obs_with(vision_dist=1.0, touch=None):
    o = np.zeros(107)
    o[70:105:5] = vision_dist  # distance column of each of the 7 rays
    if touch:
        for k, row in touch.items():
            o[10 + 6 * k:16 + 6 * k] = row
    return o


def test_curiosity_counts_vision_and_external_touch_only():
    cur = obs_with(1.0, {0: [0.5, 0, 0, 0, 0, 1], 3: [0.5, 1, 0, 0, 0, 0]})  # sensor 0 self, 3 wall
    nxt = obs_with(0.5, {0: [0.9, 0, 0, 0, 0, 1], 3: [0.9, 1, 0, 0, 0, 0]})
    pred = cur.copy()
    pred[0:10] = 5.0  # a huge proprioceptive error must not count
    err = world_prediction_error(LAYOUT, cur, nxt, pred)
    # vision: 7 rays * 0.5^2; touch: sensor 3 (0.4^2), sensors without contact 0; sensor 0 excluded (self)
    expected = (7 * 0.25 + 0.16) / (35 + 6 * 9)
    assert np.isclose(err, expected)


def test_hunger_gate_by_hand():
    e = np.array([0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0])
    np.testing.assert_allclose(hunger_gate(e, 0.8, 0.2), [0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1])


def test_hunger_silences_curiosity_but_never_the_caution_brake():
    cfg = make_config()
    assert float(cfg.drives.gate_low) == 0.2 and float(cfg.drives.set_point) == 0.8
    act = np.full((1, 1, 6), 0.5)
    for e_now, gate in ((0.15, 0.0), (0.8, 1.0), (0.95, 1.0)):
        d = Drives(LAYOUT, cfg.drives, Temperament("x", curiosity=1.0, caution=1.0), effort_cost=0.0016)
        cur, nxt = obs_with(1.0)[None, None].copy(), obs_with(0.5)[None, None].copy()
        cur[..., 105], nxt[..., 105] = e_now, e_now
        total, c = d(cur, nxt, cur.copy(), act)
        cs, ks = float(cfg.drives.curiosity_scale), float(cfg.drives.caution_scale)
        caution = -ks * 0.0016 * 0.25
        np.testing.assert_allclose(c["gate"], gate)
        # energy unchanged -> no hunger; curiosity is 1 on the first batch; caution in full at any energy
        np.testing.assert_allclose(c["weighted"]["caution"], caution)
        np.testing.assert_allclose(total, gate * cs * 1.0 + caution)
    # hunger is never gated: at E = 0.15 a meal still counts in full (no effort, no world change here)
    d = Drives(LAYOUT, cfg.drives, Temperament("x", curiosity=1.0, caution=1.0), effort_cost=0.0)
    cur, nxt = obs_with(1.0)[None, None].copy(), obs_with(1.0)[None, None].copy()
    cur[..., 105], nxt[..., 105] = 0.15, 0.45
    total, c = d(cur, nxt, cur.copy(), act)
    hunger = float(cfg.drives.hunger_scale) * ((0.8 - 0.15) ** 2 - (0.8 - 0.45) ** 2)
    np.testing.assert_allclose(c["weighted"]["hunger"], hunger)
    np.testing.assert_allclose(total, hunger)


def test_drives_combine_with_temperament_weights_and_normalize_curiosity():
    cfg = make_config()
    d = Drives(LAYOUT, cfg.drives, Temperament("x", curiosity=2.0, caution=3.0), effort_cost=0.0016)
    cur = np.stack([obs_with(1.0)] * 2)[None]
    nxt = np.stack([obs_with(0.5)] * 2)[None]
    cur[..., 105], nxt[..., 105] = 0.5, 0.6
    act = np.full((1, 2, 6), 0.5)
    total, c = d(cur, nxt, cur.copy(), act)
    np.testing.assert_allclose(c["curiosity"], 1.0)  # first batch: divided by its own mean
    hs = float(cfg.drives.hunger_scale)
    np.testing.assert_allclose(c["hunger"], 0.05 * hs)
    np.testing.assert_allclose(c["caution"], -float(cfg.drives.caution_scale) * 0.0016 * 0.25)
    # observed energy 0.5 -> gate (0.5 - 0.2) / (0.8 - 0.2) = 0.5 on curiosity only, not on hunger or caution
    np.testing.assert_allclose(c["gate"], 0.5)
    np.testing.assert_allclose(total, 0.05 * hs + 0.5 * 2.0 * float(cfg.drives.curiosity_scale) + 3.0 * c["caution"])
    w = c["weighted"]  # the contributions logged per iteration: they add up to the total
    np.testing.assert_allclose(w["hunger"], 0.05 * hs)
    np.testing.assert_allclose(w["curiosity"], 0.5 * 2.0 * float(cfg.drives.curiosity_scale))
    np.testing.assert_allclose(w["caution"], 3.0 * c["caution"])
    np.testing.assert_allclose(w["hunger"] + w["curiosity"] + w["caution"], total)


def test_protected_infancy_scales_the_cost_of_living(tmp_path, infancy):
    from umbral.body.body import BodyParams, update_energy_b
    from umbral.mind.life import Life
    from umbral.recorder import Recorder

    p = BodyParams.from_config(make_config().body)
    one = np.ones(1)
    full = update_energy_b(p, one, np.zeros(1), np.full(1, 0.5), np.zeros(1))
    low = update_energy_b(p, one, np.zeros(1), np.full(1, 0.5), np.zeros(1), metabolism=0.25)
    assert np.isclose(1 - low["after"][0], 0.25 * (1 - full["after"][0]))
    cfg = life_config(tmp_path, infancy, "life.iterations=100", "life.protected_infancy.start_factor=0.25",
                      "life.protected_infancy.fraction=0.2")
    with Recorder(cfg) as rec:
        life = Life(cfg, rec, np.random.default_rng(0), None)
    assert life.metabolism(1) == 0.25 and np.isclose(life.metabolism(11), 0.25 + 0.75 * 0.5)
    assert life.metabolism(21) == 1.0 and life.metabolism(100) == 1.0


def test_reward_scaler_matches_the_std_of_discounted_returns():
    rng = np.random.default_rng(0)
    r = rng.normal(size=(3, 50))
    s = RewardScaler(3, 0.9)
    s(r[:, :20])
    s(r[:, 20:])
    ret, all_ret = np.zeros(3), []
    for t in range(50):
        ret = 0.9 * ret + r[:, t]
        all_ret.extend(ret)
    assert np.isclose(s.std, np.std(all_ret, ddof=1))


# --- the life -------------------------------------------------------------------------

def test_life_resume_reproduces_the_continuation(tmp_path, infancy):
    full = run_experiment(life_config(tmp_path, infancy, "run.name=full", "life.run_report=false"))
    ck = full / "checkpoints" / "life_short.pt"
    assert ck.is_file() and (full / "checkpoints" / "life_long.pt").is_file()
    resumed = run_experiment(life_config(tmp_path, infancy, "run.name=resumed", "life.run_report=false",
                                         f"run.resume_from={ck.as_posix()}"))
    tail = [line for line in read_metrics(full) if json.loads(line)["step"] > 3]
    assert len(tail) > 0 and read_metrics(resumed) == tail
    a = load_checkpoint(full / "checkpoints" / "life_long.pt")
    b = load_checkpoint(resumed / "checkpoints" / "life_long.pt")
    for name, t in a.models["agent"].items():
        assert torch.equal(t, b.models["agent"][name]), name
    for k, arr in a.env["arrays"].items():
        np.testing.assert_array_equal(arr, b.env["arrays"][k], err_msg=k)
    assert a.policy["rngs"] == b.policy["rngs"]
    assert a.extra["drives"] == b.extra["drives"] and a.extra["scaler"] == b.extra["scaler"]
    assert set(a.optimizers) == {"world", "policy"}
    names = {json.loads(line)["name"] for line in read_metrics(full)}
    assert {"ppo_approx_kl", "ppo_clip_fraction", "reward_curiosity", "energy_below_0.1",
            "energy_zero_run_max"} <= names
    assert a.extra["ethics"] == b.extra["ethics"]  # the energy-0 stretches per body included
    assert "zero_cur" in a.extra["ethics"] and len(a.extra["ethics"]["zero_slot_max"]) == 4


# --- the common childhood, part 2: learning to eat ------------------------------------------

FEED = ["experiment=common_feeding", "feeding.eval_every=2", "feeding.eval_envs_per_history=1", "feeding.eval_steps=20",
        "life.protected_infancy.fraction=0.1",  # ramp span 0.6 iterations: one competent step reaches full cost
        "feeding.ramp_window=1", "feeding.ramp_max_below_01=1.01", "feeding.ramp_min_p10=0"]
EASY = ["feeding.max_below_01=1.01", "feeding.max_below_03=1.01", "feeding.min_food_per_1000=0"]
IMPOSSIBLE = ["feeding.min_food_per_1000=1e9"]


def feeding_config(tmp_path, infancy, *overrides):
    return life_config(tmp_path, infancy, *FEED, *overrides)


@pytest.fixture(scope="module")
def common_start(tmp_path_factory, infancy):
    d = run_experiment(feeding_config(tmp_path_factory.mktemp("feeding"), infancy, *EASY, "run.name=feed"))
    return d


def test_feeding_phase_is_equilibrado_over_all_histories_and_stops_at_the_criterion(common_start, infancy):
    d = common_start
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["life_status"] == "completed" and meta["feeding_phase"] is True
    assert meta["histories_of_bodies"] == ["0", "1", "2", "3"] and meta["temperament"] == "equilibrado"
    ck = load_checkpoint(d / "checkpoints" / "common_start.pt")
    assert ck.step == 2 and not (d / "checkpoints" / "life_short.pt").exists()
    rows = [json.loads(line) for line in read_metrics(d)]
    for k in ("hunger", "curiosity", "caution"):
        assert any(r["name"] == f"contrib_{k}_abs" and r["value"] > 0 for r in rows), k
    assert ck.extra["drives"]["error_mean"] is not None
    assert {f"criterion_h{h}_food_per_1000_steps" for h in "0123"} <= {r["name"] for r in rows}
    for h in "0123":  # per history, every iteration
        for k in ("energy", "energy_below_0.1", "food_per_1000_steps"):
            assert [r["step"] for r in rows if r["name"] == f"h{h}_{k}"] == [1, 2]
    assert [r["value"] for r in rows if r["name"] == "cost_factor"] == [0.25, 1.0]
    s = json.loads((d / "feeding_summary.json").read_text(encoding="utf-8"))
    assert s["status"] == "completed" and len(s["evals"]) == 1 and s["evals"][0]["iteration"] == 2
    assert set(s["evals"][0]["histories"]) == {"0", "1", "2", "3"}
    assert len(s["log_std"]["init"]) == 6 and s["log_std"]["init_mean"] == pytest.approx(-0.5)
    assert (d / "feeding_criterion.png").is_file()


def test_feeding_drives_use_the_childhood_caution_scale_or_hunger_only(tmp_path, infancy):
    from umbral.mind.life import Life
    from umbral.recorder import Recorder

    cfg = feeding_config(tmp_path, infancy, "drives.caution_scale=100", "feeding.caution_scale=50")
    with Recorder(cfg) as rec:
        life = Life(cfg, rec, np.random.default_rng(0), None, feeding=True)
    assert life.temperament.name == "equilibrado" and life.drives.caution_scale == 50.0
    with Recorder(cfg) as rec:  # a life keeps drives.caution_scale
        assert Life(cfg, rec, np.random.default_rng(0), None).drives.caution_scale == 100.0
    d = run_experiment(feeding_config(tmp_path, infancy, *EASY, "feeding.temperament=null", "run.name=hungeronly"))
    assert json.loads((d / "meta.json").read_text(encoding="utf-8"))["temperament"] == "hambre"
    rows = [json.loads(line) for line in read_metrics(d)]
    for k in ("curiosity", "caution"):  # weigh 0, but the curiosity running mean is still tracked
        assert all(r["value"] == 0.0 for r in rows if r["name"] == f"contrib_{k}_abs")
    assert load_checkpoint(d / "checkpoints" / "common_start.pt").extra["drives"]["error_mean"] is not None


def test_feeding_eval_is_pure_and_seeds_are_disjoint(tmp_path, infancy, common_start):
    from umbral.mind.life import FEEDING_SEED_BASE, Life, feeding_eval
    from umbral.recorder import Recorder

    cfg = feeding_config(tmp_path, infancy, *EASY)
    agent, _ = load_agent(load_checkpoint(common_start / "checkpoints" / "common_start.pt"))
    before = {k: v.clone() for k, v in agent.state_dict().items()}
    rng_before = torch.get_rng_state()
    a, b = feeding_eval(agent, cfg), feeding_eval(agent, cfg)
    assert a == b and torch.equal(rng_before, torch.get_rng_state())
    for k, v in agent.state_dict().items():
        assert torch.equal(v, before[k])
    with Recorder(cfg) as rec:
        life = Life(cfg, rec, np.random.default_rng(0), None, feeding=True)
        assert all(FEEDING_SEED_BASE <= life._new_seed() < FEEDING_SEED_BASE + 2**31 for _ in range(50))
        assert life.histories == ["0", "1", "2", "3"]


def test_competence_condition_by_hand():
    from umbral.mind.life import RAMP_BINS, competent

    def it(energies):
        e = np.array(energies)
        hist = np.bincount(np.clip((e * RAMP_BINS).astype(int), 0, RAMP_BINS), minlength=RAMP_BINS + 1)
        return {"steps": e.size, "low": int((e < 0.1).sum()), "hist": hist.tolist()}

    good = it([0.5] * 95 + [0.35] * 5)
    assert competent([good, good], 2, 0.01, 0.3)
    assert not competent([good], 2, 0.01, 0.3)  # the window must be full
    low = it([0.5] * 98 + [0.05] * 2)  # 2 % below 0.1 in this iteration: 1 % over the window of 2
    assert not competent([good, low], 2, 0.01, 0.3)
    assert competent([good, it([0.5] * 199 + [0.05])], 2, 0.01, 0.3)  # 1/300 < 1 %
    thin = it([0.5] * 85 + [0.25] * 15)  # p10 = 0.25 < 0.3, nobody below 0.1
    assert not competent([thin, thin], 2, 0.01, 0.3)
    assert competent([thin, thin], 2, 0.01, 0.25)  # the p10 bound is inclusive


def test_stagger_spreads_episode_ages_within_every_history():
    from umbral.mind.life import stagger

    E = 32
    one = stagger(["0"] * 32, E)
    np.testing.assert_array_equal(one, [E - (i * E) // 32 for i in range(32)])  # lives: unchanged
    hs = [h for h in "0123" for _ in range(8)]
    mixed = stagger(hs, E)
    assert sorted(mixed.tolist()) == sorted(one.tolist())  # the same slots, dealt round-robin
    for h in "0123":
        ages = sorted(mixed[[x == h for x in hs]].tolist())
        assert ages[0] <= 4 + 1 and ages[-1] >= E - 4 - 1  # every history spans the whole episode
        assert np.mean(ages) == pytest.approx(np.mean(one), abs=2)


def test_competence_ramp_rises_only_when_competent_never_falls_and_keeps_the_slope(tmp_path, infancy):
    common = [*FEED, *IMPOSSIBLE, "life.protected_infancy.fraction=0.5", "feeding.ramp_window=2"]  # span 3
    up = run_experiment(feeding_config(tmp_path, infancy, *common, "run.name=up"))
    rows = [json.loads(line) for line in read_metrics(up)]
    f = [r["value"] for r in rows if r["name"] == "cost_factor"]
    ok = [r["value"] for r in rows if r["name"] == "ramp_competent"]
    slope = 0.75 / 3
    assert f == pytest.approx([0.25, 0.25, 0.25 + slope, 0.25 + 2 * slope, 1.0, 1.0])  # window full from iteration 2
    assert ok == [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for a, b, c in zip(f, f[1:], ok):
        assert b >= a and b - a <= slope + 1e-12 and (b > a) == (c == 1.0 and a < 1.0)
    s = json.loads((up / "feeding_summary.json").read_text(encoding="utf-8"))
    assert s["full_cost_from_iteration"] == 5 and [e["iteration"] for e in s["evals"]] == [5]
    # Never competent (p10 cannot reach 1.1): the factor stays at the start, no criterion evaluation at all.
    flat = run_experiment(feeding_config(tmp_path, infancy, *common, "feeding.ramp_min_p10=1.1", "run.name=flat"))
    rows = [json.loads(line) for line in read_metrics(flat)]
    assert [r["value"] for r in rows if r["name"] == "cost_factor"] == [0.25] * 6
    s = json.loads((flat / "feeding_summary.json").read_text(encoding="utf-8"))
    assert s["status"] == "timeout" and s["evals"] == [] and s["full_cost_from_iteration"] is None
    assert s["cost_factor_final"] == 0.25 and (flat / "feeding_curves.png").is_file()


def test_feeding_phase_stops_at_the_time_cap_without_a_starting_point(tmp_path, infancy):
    d = run_experiment(feeding_config(tmp_path, infancy, *IMPOSSIBLE, "feeding.max_hours=0", "run.name=cap"))
    s = json.loads((d / "feeding_summary.json").read_text(encoding="utf-8"))
    assert s["status"] == "timeout" and s["common_start"] is None
    assert not (d / "checkpoints" / "common_start.pt").exists()


def test_feeding_resume_reproduces_the_continuation(tmp_path, infancy):
    full = run_experiment(feeding_config(tmp_path, infancy, *IMPOSSIBLE, "run.name=ffull"))
    assert json.loads((full / "meta.json").read_text(encoding="utf-8"))["life_status"] == "timeout"
    ck = full / "checkpoints" / "iter_0000002.pt"
    resumed = run_experiment(feeding_config(tmp_path, infancy, *IMPOSSIBLE, "run.name=fres",
                                            f"run.resume_from={ck.as_posix()}"))
    tail = [line for line in read_metrics(full) if json.loads(line)["step"] > 2]
    assert len(tail) > 0 and read_metrics(resumed) == tail
    a = json.loads((full / "feeding_summary.json").read_text(encoding="utf-8"))
    b = json.loads((resumed / "feeding_summary.json").read_text(encoding="utf-8"))
    assert a["evals"] == b["evals"] and len(a["evals"]) == 3 and a["log_std"] == b["log_std"]


def test_lives_start_exactly_from_the_common_starting_point(tmp_path, infancy, common_start):
    from umbral.mind.life import Life
    from umbral.recorder import Recorder

    path = common_start / "checkpoints" / "common_start.pt"
    ck = load_checkpoint(path)
    for temperament, seed in (("curioso", 3), ("cauteloso", 4)):
        cfg = life_config(tmp_path, infancy, f"agent.start={path.as_posix()}", f"agent.temperament={temperament}",
                          f"seed={seed}")
        with Recorder(cfg) as rec:
            life = Life(cfg, rec, np.random.default_rng(seed), None)
        assert life.temperament.name == temperament
        sd = life.agent.state_dict()
        assert set(sd) == set(ck.models["agent"])  # mind (with its frozen normalization), actor and critic
        for k, v in ck.models["agent"].items():
            if k == "policy.log_std":  # life.reset_action_std: back to its initial value
                assert torch.equal(sd[k], torch.full_like(v, float(cfg.agent.init_log_std))), k
            else:
                assert torch.equal(sd[k], v), k
        for name in ("world", "policy"):
            got, want = life.opts[name].state_dict()["state"], ck.optimizers[name]["state"]
            assert set(got) == set(want)
            for i in want:
                for key, v in want[i].items():
                    assert torch.equal(torch.as_tensor(got[i][key]), torch.as_tensor(v)), (name, i, key)
        assert life.drives.get_state() == ck.extra["drives"]
        fresh = RewardScaler(life.N, float(cfg.ppo.gamma))
        got, want = life.scaler.get_state(), fresh.get_state()  # the reward scaler starts afresh
        assert set(got) == set(want)
        for k in want:
            np.testing.assert_array_equal(np.asarray(got[k]), np.asarray(want[k]), err_msg=k)
        np.testing.assert_array_equal(life.scaler.ret, fresh.ret)


def test_the_action_std_is_reset_at_the_start_of_every_life_and_nothing_else(tmp_path, infancy, common_start):
    from umbral.checkpoint import save_checkpoint
    from umbral.mind.life import Life
    from umbral.recorder import Recorder

    ck = load_checkpoint(common_start / "checkpoints" / "common_start.pt")
    ck.models["agent"]["policy.log_std"] = torch.full_like(ck.models["agent"]["policy.log_std"], -2.0)
    path = save_checkpoint(tmp_path / "narrow_start.pt", ck)  # a childhood that narrowed the action std
    ck = load_checkpoint(path)
    lives = {}
    for reset in (True, False):
        cfg = life_config(tmp_path, infancy, f"agent.start={path.as_posix()}", f"life.reset_action_std={reset}",
                          f"run.name=reset{reset}")
        with Recorder(cfg) as rec:
            lives[reset] = Life(cfg, rec, np.random.default_rng(0), None)
            meta = json.loads((rec.run_dir / "meta.json").read_text(encoding="utf-8"))
        if reset:
            assert meta["action_log_std_from_start"] == [-2.0] * 6 and meta["action_log_std_reset_to"] == -0.5
        else:
            assert "action_log_std_reset_to" not in meta
    on, off = lives[True].agent.state_dict(), lives[False].agent.state_dict()
    assert torch.equal(on["policy.log_std"], torch.full((6,), -0.5))
    assert torch.equal(off["policy.log_std"], torch.full((6,), -2.0))
    for k, v in ck.models["agent"].items():  # the actor's mean, the critic and the mind stay as they were
        if k != "policy.log_std":
            assert torch.equal(on[k], v) and torch.equal(off[k], v), k
    for name in ("world", "policy"):
        a, b = lives[True].opts[name].state_dict()["state"], lives[False].opts[name].state_dict()["state"]
        for i in b:
            for key, v in b[i].items():
                assert torch.equal(torch.as_tensor(a[i][key]), torch.as_tensor(v)), (name, i, key)


def test_start_robustness_repeats_the_criterion_on_other_seed_blocks(tmp_path, infancy, common_start):
    path = common_start / "checkpoints" / "common_start.pt"
    d = run_experiment(life_config(tmp_path, infancy, *FEED, *EASY, "experiment=start_robustness",
                                   f"agent.start={path.as_posix()}", "feeding.robustness_offsets=[1000]"))
    r = json.loads((d / "robustness.json").read_text(encoding="utf-8"))
    assert [e["seed_offset"] for e in r["evals"]] == [0, 1000]
    last = json.loads((common_start / "feeding_summary.json").read_text(encoding="utf-8"))["evals"][-1]
    assert r["evals"][0]["histories"] == last["histories"]  # offset 0 is the very evaluation that passed
    assert r["evals"][1]["histories"] != last["histories"]  # another block: other worlds


def test_life_from_the_common_start_and_the_development_report(tmp_path, infancy, common_start):
    path = common_start / "checkpoints" / "common_start.pt"
    d = run_experiment(life_config(tmp_path, infancy, f"agent.start={path.as_posix()}", "run.name=fromstart"))
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["life_status"] == "completed" and meta["start"] == path.as_posix() and "infancy" not in meta
    beh = run_experiment(load_config(BASE_CONFIG, [f"run.output_dir={tmp_path.as_posix()}",
                                                   "experiment=behavior_battery", f"behavior.checkpoint={path.as_posix()}",
                                                   "behavior.history=0", *TINY_LIFE]))
    assert json.loads((beh / "meta.json").read_text(encoding="utf-8"))["temperament"] == "equilibrado"
    rep = run_experiment(load_config(BASE_CONFIG, [f"run.output_dir={tmp_path.as_posix()}", "experiment=agent_report",
                                                   f"report.agents=[{d.as_posix()}]", "behavior.n_boot=20",
                                                   f"report.start_behavior={beh.as_posix()}"]))
    report = json.loads((rep / "report.json").read_text(encoding="utf-8"))
    dev = next(iter(report["development"].values()))
    assert set(dev["means"]["effort"]) == {"start", "short", "long"}
    ez = report["energy_zero"]
    assert ez["start"]["episodes"] == 2
    assert next(v for k, v in ez.items() if k != "start")["behavior"]["episodes"] == 2


def test_life_report_writes_behavior_and_prediction_check(tmp_path, infancy):
    d = run_experiment(life_config(tmp_path, infancy, "run.name=report"))
    beh = json.loads((d / "behavior.json").read_text(encoding="utf-8"))
    for m in ("distance", "area", "food", "effort", "neutral_contacts", "probe_delta", "probe_latency"):
        assert len(beh[m]) == 2
    assert all(n >= 1 for n in beh["probes"])
    for z, run, low in zip(beh["energy_zero"], beh["energy_zero_run"], beh["energy_below_0.1"]):
        assert 0.0 <= z <= low and 0 <= run <= z * 60 + 1e-9
    pc = json.loads((d / "prediction_check.json").read_text(encoding="utf-8"))
    assert set(pc["channels"]) == {"proprio", "touch", "vision", "intero"}
    own = json.loads((d / "prediction_own.json").read_text(encoding="utf-8"))  # on the agent's own behaviour
    assert own["behaviour"] == "own" and set(own["channels"]) == set(pc["channels"])
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert "would_flag_at_iteration" in meta["ethics"]["zero_runs"]
    assert meta["life_status"] == "completed" and "fraction_below_0.1" in meta["ethics"]
    short = json.loads((d / "behavior_short.json").read_text(encoding="utf-8"))
    assert short["checkpoint"].endswith("life_short.pt") and len(short["effort"]) == 2
    # phases: 6 iterations, protected childhood = round(0.2 * 6) = 1 iteration
    ph = meta["ethics"]["phases"]
    assert ph["infancy"]["steps"] == 4 * 16 and ph["adult"]["steps"] == 5 * 4 * 16
    for v in ph.values():  # the bodies one by one, not only their mean
        assert 0.0 <= v["energy_min"] - 0.01 <= v["energy_p10"] <= v["mean_energy"] + 0.1  # p10 at 0.01 resolution
    assert {"energy_min", "energy_p10"} <= {json.loads(line)["name"] for line in read_metrics(d)}
    rows = [json.loads(line) for line in read_metrics(d)]
    shares = [r for r in rows if r["name"].startswith("share_")]
    assert len(shares) == 3 * 6
    gates = [r for r in rows if r["name"] == "gate_mean"]
    assert len(gates) == 6 and all(0.0 <= r["value"] <= 1.0 for r in gates)
    for step in range(1, 7):
        assert np.isclose(sum(r["value"] for r in shares if r["step"] == step), 1.0)
    rep = run_experiment(load_config(BASE_CONFIG, [f"run.output_dir={tmp_path.as_posix()}", "experiment=agent_report",
                                                   f"report.agents=[{d.as_posix()}]", "behavior.n_boot=20"]))
    report = json.loads((rep / "report.json").read_text(encoding="utf-8"))
    info = next(iter(report["agents"].values()))
    assert info["drive_shares"]["infancy"]["iterations"] == 1 and info["drive_shares"]["adult"]["iterations"] == 5
    assert report["behavior"] and report["behavior_short"]
    assert (rep / "behavior.png").is_file() and (rep / "behavior_short.png").is_file()


def test_ethics_rule_flags_and_stops(tmp_path, infancy):
    d = run_experiment(life_config(tmp_path, infancy, "run.name=flag", "ethics.low_energy=1.01",
                                   "ethics.warmup_fraction=0.0"))
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["life_status"] == "flagged"
    summary = json.loads((d / "flagged_summary.json").read_text(encoding="utf-8"))
    assert {"energy_curve", "fraction_below_0.1", "fraction_below_0.3", "food_eaten"} <= set(summary)
    assert not (d / "checkpoints" / "life_long.pt").exists()


# --- the population ---------------------------------------------------------------------

def test_population_runs_resumes_and_skips_done_agents(tmp_path, infancy):
    pop = tmp_path / "pop"
    over = json.dumps([*TINY_LIFE, f"agent.infancy={infancy}", "life.run_report=false"])

    def session(*extra):
        cfg = load_config(BASE_CONFIG, [f"run.output_dir={tmp_path.as_posix()}", "experiment=train_population",
                                        f"population.dir={pop.as_posix()}", "population.temperaments=[curioso, cauteloso]",
                                        "population.histories=['0']", "population.seeds=[0]",
                                        f"population.overrides={over}", *extra])
        run_experiment(cfg)
        return json.loads((pop / "population.json").read_text(encoding="utf-8"))["agents"]

    first = session("population.max_agents=1")
    a, b = "curioso_h0_s0", "cauteloso_h0_s0"
    assert first[a]["status"] == "done" and first[b]["status"] == "pending"

    # Interrupt b half-way: a paused run (stop_at) in the population folder, recorded as partial.
    paused = run_experiment(load_config(str(pop / "config.yaml"), [
        "experiment=live_agent", f"run.output_dir={(pop / 'agents').as_posix()}", f"run.name={b}",
        f"seed={first[b]['seed']}", "agent.temperament=cauteloso", "agent.history=0", *json.loads(over),
        "life.stop_at=3"]))
    index = json.loads((pop / "population.json").read_text(encoding="utf-8"))
    index["agents"][b].update(status="partial", runs=[str(paused)])
    (pop / "population.json").write_text(json.dumps(index), encoding="utf-8")

    second = session()
    assert second[a] == first[a]  # skipped: untouched
    assert second[b]["status"] == "done" and len(second[b]["runs"]) == 2
    meta = json.loads((__import__("pathlib").Path(second[b]["runs"][1]) / "meta.json").read_text(encoding="utf-8"))
    assert meta["resumed_at_step"] == 3


# --- energy-0 stretches body by body, recorded battery, prediction on own behaviour ----------------

def test_energy_zero_stretches_are_counted_body_by_body(tmp_path, infancy):
    from umbral.mind.life import Life
    from umbral.recorder import Recorder

    cfg = life_config(tmp_path, infancy, "run.name=zr")
    with Recorder(cfg) as rec:
        life = Life(cfg, rec, np.random.default_rng(0), None)
        life._reset_acc()
        e = np.full((4, 16), 0.5)
        e[0, 2:6] = 0.0     # a stretch of 4 that ends inside the iteration
        e[1, 10:] = 0.0     # 6 steps, still open at the end of the iteration ...
        e[2, :] = 0.0       # 16 steps, but its episode restarts: closed
        e[3, 3] = 1e-9      # not exactly 0: no stretch
        life.life_left[:] = [3, 3, 0, 3]
        life._update_zero_runs(it=1, e=e, phase="infancy")
        ex = life.ethics
        assert ex["zero_cur"] == [0, 6, 0, 0] and ex["zero_slot_max"] == [4, 6, 16, 0]
        assert ex["zero_runs"] == 2 and ex["zero_run_longest"] == 16
        e2 = np.full((4, 16), 0.5)
        e2[1, :3] = 0.0     # ... it continues: 6 + 3 = 9
        life.life_left[:] = [2, 2, 3, 2]
        life._update_zero_runs(it=2, e=e2, phase="infancy")
        assert life.ethics["zero_slot_max"] == [4, 9, 16, 0] and life.ethics["zero_runs"] == 3
        z = life.zero_run_summary()
        assert z["longest"] == 16 and z["stretches"] == 3 and z["ge200"] == 0
        assert life.acc["energy_zero_run_max"] == 16 + 9


def test_the_individual_rule_only_observes(tmp_path, infancy):
    from umbral.mind.life import Life
    from umbral.recorder import Recorder

    cfg = life_config(tmp_path, infancy, "run.name=zobs", "ethics.zero_run_thresholds=[3,10]")
    with Recorder(cfg) as rec:
        life = Life(cfg, rec, np.random.default_rng(0), None)
        life._reset_acc()
        life.life_left[:] = 3
        e = np.full((4, 16), 0.5)
        e[0, 4:9] = 0.0  # 5 steps: >= 3, < 10
        life._update_zero_runs(it=0, e=e, phase="infancy")  # warm-up: observed but not a "would stop"
        assert life.zero_run_summary()["would_flag_at_iteration"] == {"3": None, "10": None}
        life._update_zero_runs(it=5, e=e, phase="infancy")
        z = life.zero_run_summary()
        assert z["would_flag_at_iteration"] == {"3": 5, "10": None} and z["ge3"] == 2 and z["ge3_after_warmup"] == 1
    d = run_experiment(life_config(tmp_path, infancy, "run.name=zobs2", "life.run_report=false",
                                   "ethics.zero_run_thresholds=[1]", "body.energy.initial=0.0"))
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["life_status"] != "flagged" or meta["ethics"]["fraction_below_0.1_after_warmup"] > 0.10
    assert "would_flag_at_iteration" in meta["ethics"]["zero_runs"]  # computed, reported, stops nothing


def test_recording_a_battery_does_not_change_it(tmp_path, infancy, common_start):
    from umbral.metrics.behavior import run_battery

    cfg = life_config(tmp_path, infancy)
    agent, _ = load_agent(load_checkpoint(common_start / "checkpoints" / "common_start.pt"))
    plain = run_battery(agent, cfg, "0")
    recorded = run_battery(agent, cfg, "0", record=[1])
    sig = recorded.pop("recordings")[1]
    assert json.dumps(plain, sort_keys=True) == json.dumps(recorded, sort_keys=True)
    T = int(cfg.behavior.steps)
    assert sig["energy"].shape == (T,) and sig["trunk_pose"].shape == (T, 3) and sig["action_mean"].shape == (T, 6)
    assert sig["neutral_pos"].shape[0] == T and sig["neutral_pos"].shape[1] == len(sig["neutral_r"])
    assert np.isclose(sig["food_eaten"].sum(), plain["food"][1])


def test_battery_replay_matches_the_battery_and_writes_viewer_signals(tmp_path, infancy, common_start):
    from umbral.metrics.behavior import run_battery

    path = common_start / "checkpoints" / "common_start.pt"
    d = run_experiment(life_config(tmp_path, infancy, "experiment=battery_replay", f"behavior.checkpoint={path.as_posix()}",
                                   "behavior.history=0", "replay.episodes=[0,1]", "run.name=replay"))
    r = json.loads((d / "replay.json").read_text(encoding="utf-8"))
    agent, _ = load_agent(load_checkpoint(path))
    battery = run_battery(agent, life_config(tmp_path, infancy), "0")
    assert r["measures"]["energy_zero"] == battery["energy_zero"] and r["measures"]["food"] == battery["food"]
    for i in ("0", "1"):
        assert r["episodes"][i]["battery"]["effort"] == battery["effort"][int(i)]
        sig = np.load(d / f"signals_ep{i}.npz")
        assert {"trunk_pose", "limb_points", "food_pos", "food_active", "neutral_pos", "arena_size"} <= set(sig.files)


def test_zero_episode_analysis_on_a_synthetic_recording():
    from umbral.experiments.agent import analyze_zero_episode, zero_runs

    assert zero_runs(np.array([0.5, 0, 0, 0.1, 0, 0.2, 0, 0])) == [(1, 3), (4, 5), (6, 8)]
    T = 10
    e = np.array([0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.3, 0.5, 0.6, 0.6])
    sig = {"energy": e, "trunk_pose": np.c_[np.arange(T) * 0.1, np.zeros(T), np.zeros(T)],
           "action_executed": np.full((T, 6), 0.3), "action_intended": np.full((T, 6), 1.0),
           "action_mean": np.full((T, 6), 0.9), "food_pos": np.tile([[5.0, 0.0]], (T, 1, 1)),
           "food_active": np.ones((T, 1), bool), "arena_size": np.float32(10), "obstacle_pos": np.zeros((0, 2)),
           "obstacle_r": np.zeros(0), "raw_flat": np.zeros((T, 1)), "contacts": np.zeros(T),
           "food_eaten": np.r_[np.zeros(6), 1, np.zeros(3)], "env_seed": np.int64(7), "history_id": np.array("2")}
    a = analyze_zero_episode(sig)
    assert a["longest_run"] == 4 and a["zero_runs"] == [(2, 6)] and a["first_below_low"] == 2
    L = a["longest"]
    assert L["start"] == 2 and L["end"] == 6 and L["ends_by"] == "food"
    assert np.isclose(L["speed"]["during"], 0.1) and np.isclose(L["nearest_food"]["during"], 5 - 0.35)


def test_evaluation_with_the_agents_own_policy(tmp_path, infancy, common_start):
    path = common_start / "checkpoints" / "common_start.pt"
    d = run_experiment(life_config(tmp_path, infancy, "experiment=prediction_own", f"prediction.checkpoints=[{path.as_posix()}]",
                                   "eval.n_boot=50", "run.name=predown"))
    r = json.loads((d / "prediction_own.json").read_text(encoding="utf-8"))["checkpoints"][path.as_posix()]
    assert r["ou"]["behaviour"] == "ou" and r["own"]["behaviour"] == "own"
    from umbral.experiments.agent import prediction_check

    cfg = life_config(tmp_path, infancy, "eval.n_boot=50")
    agent, _ = load_agent(load_checkpoint(path))
    again = prediction_check(agent.world, cfg.agent.infancy, cfg, agent=agent)  # pure: same numbers again
    for c, v in r["own"]["channels"].items():
        assert v["agent"] == again["channels"][c]["agent"] and v["childhood"] == again["channels"][c]["childhood"]
        assert v["agent"] != r["ou"]["channels"][c]["agent"]  # other data: the agent's own behaviour
        assert set(v["by_history"]) == {"0", "1", "2", "3"}


def test_the_individual_rule_stops_the_life_at_the_limit(tmp_path, infancy):
    # Bodies start with no energy: a stretch at 0 starts at once. Limit of 5 steps, warm-up 0.3 iterations.
    d = run_experiment(life_config(tmp_path, infancy, "run.name=zlim", "life.run_report=false",
                                   "body.energy.initial=0.0", "ethics.zero_run_limit=5"))
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["life_status"] == "flagged" and meta["flag_reason"] == "zero_run"
    z = meta["ethics"]["zero_runs"]
    assert z["flagged_at_iteration"] == 1 and z["longest"] >= 5 and meta["ethics"]["zero_run_limit"] == 5
    assert (d / "flagged_summary.json").is_file() and (d / "checkpoints" / "iter_0000001.pt").is_file()
    assert "individual rule" in (d / "log.txt").read_text(encoding="utf-8")


def test_without_a_limit_the_individual_rule_only_observes(tmp_path, infancy):
    d = run_experiment(life_config(tmp_path, infancy, "run.name=znolim", "life.run_report=false",
                                   "body.energy.initial=0.0", "ethics.zero_run_limit=null", "ethics.max_low_fraction=1.0"))
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["life_status"] == "completed" and meta["flag_reason"] is None
    assert meta["ethics"]["zero_runs"]["flagged_at_iteration"] is None


def test_the_battery_marks_episodes_for_review(tmp_path, infancy):
    from umbral.experiments.agent import battery_review

    cfg = life_config(tmp_path, infancy, "ethics.zero_run_limit=200")
    assert battery_review({"energy_zero_run": [0.0, 199.0, 200.0, 1404.0]}, cfg) == [2, 3]
    assert battery_review({"energy_zero_run": [1404.0]}, life_config(tmp_path, infancy, "ethics.zero_run_limit=null")) == []


def test_zero_diagnosis_reports_the_drives_at_energy_zero(tmp_path, infancy):
    # A short life whose bodies start with no energy; then the diagnosis from its checkpoint.
    life = run_experiment(life_config(tmp_path, infancy, "run.name=zdiag_life", "life.run_report=false",
                                      "body.energy.initial=0.0", "ethics.zero_run_limit=null",
                                      "ethics.max_low_fraction=1.0", "life.stop_at=2"))
    ck = life / "checkpoints" / "iter_0000002.pt"
    d = run_experiment(life_config(tmp_path, infancy, "experiment=zero_diagnosis", f"diagnosis.checkpoint={ck.as_posix()}",
                                   "diagnosis.steps=40", "ethics.zero_run_limit=5", "run.name=zdiag"))
    r = json.loads((d / "diagnosis.json").read_text(encoding="utf-8"))
    z = r["bands"]["zero"]
    assert z["steps"] > 0 and z["hunger"] == 0.0 and z["curiosity"] == 0.0 and z["gate"] == 0.0  # no drive but caution
    assert z["caution"] <= 0.0 and z["total"] == z["caution"]
    assert "infancy" in r["training_exposure"] and isinstance(r["zero_stretches"], list)
    again = run_experiment(life_config(tmp_path, infancy, "experiment=zero_diagnosis", f"diagnosis.checkpoint={ck.as_posix()}",
                                       "diagnosis.steps=40", "ethics.zero_run_limit=5", "run.name=zdiag2"))
    assert (again / "diagnosis.json").read_text(encoding="utf-8").replace(str(again), "") == \
        (d / "diagnosis.json").read_text(encoding="utf-8").replace(str(d), "")  # deterministic from the checkpoint
