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
TINY_LIFE = ["mind.encoder=16", "mind.hidden=16", "mind.head_hidden=16", "agent.width=16",
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
    assert {"ppo_approx_kl", "ppo_clip_fraction", "reward_curiosity", "energy_below_0.1"} <= names


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
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
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
