import json

import numpy as np
import torch

from conftest import BASE_CONFIG, make_config
from umbral.body.sensors import ObsLayout
from umbral.cli import run_experiment
from umbral.config import load_config
from umbral.mind.model import CHANNELS, build_model, channel_sq_error, n_parameters

LAYOUT = ObsLayout(7)


def models():
    cfg = make_config()
    torch.manual_seed(0)
    return build_model(LAYOUT, cfg.mind, True), build_model(LAYOUT, cfg.mind, False)


def batch(B=3, T=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(B, T, 107, generator=g), torch.rand(B, T, 6, generator=g) * 2 - 1,
            torch.rand(B, T, 6, generator=g) * 2 - 1)


def test_size_and_shapes():
    eff, _ = models()
    assert n_parameters(eff) < 500_000
    obs, prev, act = batch()
    out = eff(obs, prev, act)
    assert out.hidden.shape == (3, 5, 256) and out.h_last.shape == (1, 3, 256)
    assert out.touch_logits.shape == (3, 5, 10, 6) and out.vision_logits.shape == (3, 5, 7, 5)
    assert torch.equal(out.hidden[:, -1], out.h_last[0])
    assert eff.predict_next(obs, out).shape == obs.shape
    losses = eff.loss(obs[:, :-1], obs[:, 1:], eff(obs[:, :-1], prev[:, :-1], act[:, :-1]))
    assert set(losses) == {*CHANNELS, "total"}
    assert torch.isclose(losses["total"], sum(losses[c] for c in CHANNELS))


def test_efference_copy_matters_only_in_the_efference_model():
    eff, noeff = models()
    obs, prev, act = batch()
    other_prev, other_act = -prev, -act
    with torch.no_grad():
        assert not torch.equal(eff(obs, prev, act).reg, eff(obs, other_prev, other_act).reg)
        assert torch.equal(noeff(obs, prev, act).reg, noeff(obs, other_prev, other_act).reg)
        # The GRU sees a_{t-1}; the head sees a_t.
        assert torch.equal(eff(obs, prev, act).hidden, eff(obs, prev, other_act).hidden)
        assert not torch.equal(eff(obs, prev, act).hidden, eff(obs, other_prev, act).hidden)


def test_hidden_states_carry_across_segments():
    eff, _ = models()
    obs, prev, act = batch(T=8)
    with torch.no_grad():
        whole = eff(obs, prev, act)
        first = eff(obs[:, :4], prev[:, :4], act[:, :4])
        second = eff(obs[:, 4:], prev[:, 4:], act[:, 4:], first.h_last)
    torch.testing.assert_close(torch.cat([first.hidden, second.hidden], 1), whole.hidden)


def test_delta_std_is_part_of_the_state_dict():
    eff, _ = models()
    eff.set_delta_std(np.arange(1, 30, dtype=float))
    assert torch.equal(eff.state_dict()["delta_std"], torch.arange(1, 30, dtype=torch.float32))


def test_persistence_error_is_zero_for_identical_steps():
    obs, _, _ = batch()
    for s, n in channel_sq_error(LAYOUT, obs, obs).values():
        assert s == 0.0 and n > 0


def test_training_logs_both_models_and_shares_normalization(tmp_path):
    from test_checkpoint_world import TINY

    cfg = load_config(BASE_CONFIG, [f"run.output_dir={tmp_path.as_posix()}", "experiment=train_world_model", *TINY])
    d = run_experiment(cfg)
    names = {json.loads(line)["name"] for line in (d / "metrics.jsonl").read_text(encoding="utf-8").splitlines()}
    for p in ("efference", "no_efference", "persistence"):
        for c in CHANNELS:
            assert f"{p}/mse_{c}" in names
    from umbral.checkpoint import load_checkpoint

    ck = load_checkpoint(d / "checkpoints" / "update_0000006.pt")
    assert torch.equal(ck.models["efference"]["delta_std"], ck.models["no_efference"]["delta_std"])
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert set(meta["time_fractions"]) == {"world", "model_efference", "model_no_efference", "metrics_and_io"}
    assert (d / "learning_curves.png").is_file() and (d / "checkpoint_eval_curves.png").is_file()
    assert any(n.startswith("ckpt_eval/efference/mse_") for n in names)

    ev = load_config(BASE_CONFIG, [f"run.output_dir={tmp_path.as_posix()}", "experiment=eval_world_model",
                                   f"eval.checkpoint={(d / 'checkpoints' / 'update_0000006.pt').as_posix()}",
                                   "eval.envs_per_history=1", "eval.steps=20", "eval.chunk=8"])
    e = run_experiment(ev)
    res = json.loads((e / "eval_results.json").read_text(encoding="utf-8"))
    assert set(res["mse"]) == set(CHANNELS) and (e / "learning_curves.png").is_file()
    assert set(res["touch_events"]) == {"onset", "onset_self", "onset_external", "offset", "offset_self",
                                        "offset_external"}
    assert "touch_onset_self_beats_no_efference" in res["criteria"]


def test_bench_env_records_throughput(tmp_path):
    cfg = load_config(BASE_CONFIG, [f"run.output_dir={tmp_path.as_posix()}", "experiment=bench_env",
                                    "bench.sizes=[1, 4]", "bench.seconds=0.2"])
    d = run_experiment(cfg)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert set(meta["env_steps_per_second"]) == {"1", "4"}


def test_touch_event_errors_select_the_right_sensor_steps():
    from umbral.mind.evaluation import touch_event_sq_error

    cur = torch.zeros(1, 4, 107)
    nxt = torch.zeros(1, 4, 107)
    t0 = 10
    def row(k):  # columns of touch sensor k
        return slice(t0 + 6 * k, t0 + 6 * (k + 1))
    # step 0: sensor 8 (left tip) starts touching itself (self-contact onset)
    nxt[0, 0, row(8)] = torch.tensor([0.7, 0, 0, 0, 0, 1])
    # step 1: sensor 2 starts touching an obstacle (external onset)
    nxt[0, 1, row(2)] = torch.tensor([1.0, 0, 0, 1, 0, 0])
    # step 2: sensor 3 stops touching a wall (external offset); sensor 9 keeps touching (no event)
    cur[0, 2, row(3)] = torch.tensor([0.5, 1, 0, 0, 0, 0])
    cur[0, 2, row(9)] = torch.tensor([0.4, 0, 1, 0, 0, 0])
    nxt[0, 2, row(9)] = torch.tensor([0.6, 0, 1, 0, 0, 0])
    pred = cur.clone()  # persistence-like prediction
    res = touch_event_sq_error(LAYOUT, cur, nxt, pred)
    assert res["onset"][1] == 2 and res["onset_self"][1] == 1 and res["onset_external"][1] == 1
    assert res["offset"][1] == 1 and res["offset_external"][1] == 1 and res["offset_self"][1] == 0
    assert np.isclose(res["onset_self"][0], (0.7**2 + 1) / 6)
    assert np.isclose(res["offset_external"][0], (0.5**2 + 1) / 6)
    perfect = touch_event_sq_error(LAYOUT, cur, nxt, nxt)
    assert all(s == 0.0 for s, _ in perfect.values())


def test_bootstrap_ci_is_stratified_and_brackets_the_estimate():
    from umbral.mind.evaluation import bootstrap_ratio_ci

    rng = np.random.default_rng(0)
    hist = ["0"] * 10 + ["1"] * 10
    den = np.stack([rng.uniform(1, 2, 20), np.full(20, 10.0)], 1)
    num = den.copy()
    num[:, 0] *= rng.uniform(0.5, 0.9, 20)
    lo, hi = bootstrap_ratio_ci(num, den, hist, 2000, 1)
    point = (num[:, 0].sum() / num[:, 1].sum()) / (den[:, 0].sum() / den[:, 1].sum())
    assert lo < point < hi and 0.5 < lo and hi < 0.9
    assert bootstrap_ratio_ci(den, den, hist, 200, 1) == (1.0, 1.0)
    assert bootstrap_ratio_ci(num, den, hist, 500, 3) == bootstrap_ratio_ci(num, den, hist, 500, 3)
