"""The energy debt of hunger (25 Nov 2026, docs/protocolo.md).

With the observed energy alone, hunger stopped at 0: ``D`` could not rise any more and
reaching 0 worked as a relief. The internal energy ``E_int`` keeps falling at 0 by the
body's cost, so the signal goes on; the body and the observation do not change.
"""

import json

import numpy as np

from conftest import make_config, read_metrics
from test_agent import LAYOUT, infancy, life_config, obs_with  # noqa: F401  (fixture)
from umbral.checkpoint import load_checkpoint
from umbral.cli import run_experiment
from umbral.mind.drives import Drives, Temperament, homeostatic_drive, hunger_reward, internal_energy_step

CFG = make_config()
SP, HS = float(CFG.drives.set_point), float(CFG.drives.hunger_scale)
FLOOR = float(CFG.drives.debt_floor)


def body_energy(e0: float, costs: np.ndarray, food: dict[int, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Physical energy (clipped at 0) and E_int of one body over a sequence of step costs."""
    food = food or {}
    e, ei = [e0], [e0]
    for t, c in enumerate(costs):
        nxt = min(1.0, max(0.0, e[-1] - c + food.get(t, 0.0)))
        e.append(nxt)
        ei.append(float(internal_energy_step(np.array([ei[-1]]), np.array([nxt]), np.array([c]),
                                             np.array([t in food]), FLOOR)[0]))
    return np.array(e), np.array(ei)


def drives_hunger(e_obs: np.ndarray, e_int: np.ndarray | None, debt: bool = True) -> np.ndarray:
    cfg = make_config(f"drives.hunger_debt={str(debt).lower()}")
    d = Drives(LAYOUT, cfg.drives, Temperament("x", curiosity=0.0, caution=0.0), effort_cost=0.0)
    o = np.stack([obs_with(1.0)] * len(e_obs))[None].copy()
    o[..., 105] = e_obs
    _, c = d(o[:, :-1], o[:, 1:], o[:, 1:].copy(), np.zeros((1, len(e_obs) - 1, 6)),
             None if e_int is None else e_int[None])
    return c["weighted"]["hunger"][0]


def D(e):
    """The drive by hand: the parabola above 0, its tangent at 0 below."""
    e = np.asarray(e, float)
    return np.where(e >= 0, (SP - e) ** 2, SP ** 2 - 2 * SP * e)


def test_below_zero_the_drive_is_the_tangent_at_zero():
    e = np.array([0.5, 0.0, -1e-9, -0.1, -1.0])
    np.testing.assert_array_equal(homeostatic_drive(e, SP), D(e))
    h = 1e-7  # continuous, with the same slope on both sides of 0
    left, right = (homeostatic_drive(0.0, SP) - homeostatic_drive(-h, SP)) / h, (
        homeostatic_drive(h, SP) - homeostatic_drive(0.0, SP)) / h
    np.testing.assert_allclose(left, -2 * SP, rtol=1e-8)
    np.testing.assert_allclose(right, -2 * SP, rtol=1e-5)


def test_the_debt_is_on_by_default_with_its_floor():
    assert CFG.drives.hunger_debt is True and FLOOR == -1.0


def test_with_energy_above_zero_hunger_is_identical_to_before():
    rng = np.random.default_rng(0)
    costs = rng.uniform(0.0005, 0.003, 300)
    food = {int(t): 0.3 for t in rng.choice(300, 6, replace=False)}
    e, ei = body_energy(0.9, costs, food)
    assert (e > 0).all()
    np.testing.assert_array_equal(ei, e)  # E_int is exactly the observed energy
    np.testing.assert_array_equal(drives_hunger(e, ei), drives_hunger(e, None, debt=False))


def test_at_zero_the_signal_goes_on_as_just_before_and_never_vanishes():
    c = 0.0012
    e, ei = body_energy(0.0050, np.full(40, c))
    zero = np.flatnonzero(e[:-1] == 0.0)  # steps that start at physical energy 0
    first = int(np.flatnonzero(e == 0.0)[0])  # the step that reaches 0
    assert len(zero) > 30
    v1 = drives_hunger(e, None, debt=False)
    debt = drives_hunger(e, ei)
    assert (v1[zero] == 0.0).all()  # before: no signal at all at 0 (the false relief)
    assert (debt[zero] < 0.0).all()  # now: it goes on
    np.testing.assert_allclose(debt[:first - 1], v1[:first - 1])  # identical while E > 0
    np.testing.assert_allclose(debt, HS * (D(ei[:-1]) - D(ei[1:])))
    # in debt the reward per step is constant: the slope of D at 0 times the cost, however deep the debt
    np.testing.assert_allclose(debt[zero], -HS * 2 * SP * c, rtol=1e-9)
    assert ei[-1] < -0.04
    # the same as the last whole step before 0 with the same cost, but for the parabola's curvature there
    # (HS * c * (2 E - c) with E < 2c: at most 1e-3 of the step here); with the parabola it grew with the debt
    before = debt[first - 2]
    assert abs(debt[zero] - before).max() <= HS * c * (2 * e[first - 2] - c) * (1 + 1e-9)
    parabola = HS * ((SP - ei[:-1]) ** 2 - (SP - ei[1:]) ** 2)
    assert (np.diff(parabola[zero]) < 0).all() and np.ptp(debt[zero]) < 1e-12
    np.testing.assert_allclose(ei[-1], 0.0050 - 40 * c)  # the debt accumulates what the body would have spent


def test_eating_at_zero_settles_the_debt():
    c = 0.002
    costs = np.full(200, c)
    e, ei = body_energy(0.01, costs, {150: 0.3})
    debt_before = ei[150]
    assert e[150] == 0.0 and debt_before < -0.2
    assert ei[151] == e[151] and np.isclose(e[151], 0.3 - c)  # the food settles the debt: back to the physical energy
    h = drives_hunger(e, ei)
    np.testing.assert_allclose(h[150], HS * (D(debt_before) - D(e[151])))
    assert h[150] > drives_hunger(e, None, debt=False)[150] > 0  # worth more than the meal from 0 was before
    assert (ei[151:] == e[151:]).all()


def test_the_debt_stops_at_its_floor():
    e, ei = body_energy(0.0, np.full(800, 0.002))
    assert ei.min() == FLOOR and (ei >= FLOOR).all()
    at = np.flatnonzero(ei == FLOOR)
    assert at[0] in (500, 501) and (np.diff(at) == 1).all() and at[-1] == 800
    h = drives_hunger(e, ei)
    assert (h[at[0]:] == 0.0).all()  # at the floor the signal stops: that is why bodies there are reported


def test_a_life_keeps_its_debt_in_the_checkpoint_and_resumes_it(tmp_path, infancy):
    common = ["life.run_report=false", "body.energy.initial=0.0", "ethics.zero_run_limit=null",
              "ethics.max_low_fraction=1.0"]
    full = run_experiment(life_config(tmp_path, infancy, "run.name=debt_full", *common))
    a = load_checkpoint(full / "checkpoints" / "life_long.pt")
    short = full / "checkpoints" / "life_short.pt"
    s = load_checkpoint(short)
    assert min(s.extra["e_int"]) < 0.0  # bodies at 0 are in debt when the checkpoint is taken
    resumed = run_experiment(life_config(tmp_path, infancy, "run.name=debt_resumed", *common,
                                         f"run.resume_from={short.as_posix()}"))
    b = load_checkpoint(resumed / "checkpoints" / "life_long.pt")
    assert a.extra["e_int"] == b.extra["e_int"] and a.extra["ethics"] == b.extra["ethics"]
    tail = [line for line in read_metrics(full) if json.loads(line)["step"] > 3]
    assert read_metrics(resumed) == tail
    m = {json.loads(line)["name"] for line in read_metrics(full)}
    assert {"hunger_debt_min", "hunger_debt_steps"} <= m
    meta = json.loads((full / "meta.json").read_text(encoding="utf-8"))
    d = meta["ethics"]["hunger_debt"]
    assert d["enabled"] and d["min"] < 0.0 and d["floor"] == -1.0 and d["floor_steps"] == 0


def test_hunger_v1_stays_available_and_silent_at_zero(tmp_path, infancy):
    d = run_experiment(life_config(tmp_path, infancy, "run.name=v1", "life.run_report=false", "body.energy.initial=0.0",
                                   "ethics.zero_run_limit=null", "ethics.max_low_fraction=1.0", "drives.hunger_debt=false",
                                   "life.stop_at=2"))
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["ethics"]["hunger_debt"]["enabled"] is False


def test_bodies_at_the_floor_are_reported(tmp_path, infancy):
    d = run_experiment(life_config(tmp_path, infancy, "run.name=floor", "body.energy.initial=0.0",
                                   "ethics.zero_run_limit=null", "ethics.max_low_fraction=1.0", "drives.debt_floor=-0.001",
                                   "life.run_report=false"))
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    f = meta["ethics"]["hunger_debt"]
    assert f["floor_steps"] > 0 and f["floor_bodies"] > 0 and f["floor_first_iteration"] >= 1
    assert "DEBT FLOOR" in (d / "log.txt").read_text(encoding="utf-8")
