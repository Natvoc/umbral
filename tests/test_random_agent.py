import json

import numpy as np

from conftest import BASE_CONFIG, read_metrics
from umbral.cli import run_experiment
from umbral.config import load_config


def run(tmp_path, *overrides):
    cfg = load_config(BASE_CONFIG, [f"run.output_dir={tmp_path.as_posix()}", "experiment=random_agent",
                                    "random_agent.steps=120", "random_agent.gif_steps=20", *overrides])
    return run_experiment(cfg)


def test_random_agent_leaves_complete_run(tmp_path):
    d = run(tmp_path)
    for name in ("config.yaml", "meta.json", "metrics.jsonl", "log.txt", "signals.npz", "run.gif"):
        assert (d / name).is_file(), name
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "completed" and meta["env_steps_per_second"] > 0
    names = {json.loads(line)["name"] for line in read_metrics(d)}
    assert names == {"energy", "food_eaten", "contacts", "void_action"}
    with np.load(d / "signals.npz") as z:
        assert z["obs_flat"].shape == (120, 107)
        assert z["touch"].shape == (120, 10, 6) and z["vision"].shape == (120, 7, 5)
        assert z["action_intended"].shape == (120, 6) and z["limb_points"].shape == (120, 2, 3, 2)
        np.testing.assert_array_equal(z["obs_flat"], z["raw_flat"])  # link intact


def test_random_agent_is_reproducible(tmp_path):
    a = run(tmp_path, "seed=4", "run.name=a")
    b = run(tmp_path, "seed=4", "run.name=b")
    assert (a / "metrics.jsonl").read_bytes() == (b / "metrics.jsonl").read_bytes()
    with np.load(a / "signals.npz") as za, np.load(b / "signals.npz") as zb:
        np.testing.assert_array_equal(za["obs_flat"], zb["obs_flat"])
