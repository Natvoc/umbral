import json

from conftest import read_metrics
from umbral.experiments.noop import checkpoint_path


def test_resume_reproduces_second_half(run_noop):
    full = run_noop("seed=3", "run.name=full")
    ckpt = checkpoint_path(full, 100)
    assert ckpt.is_file()

    resumed = run_noop("seed=3", "run.name=resumed", f"run.resume_from={ckpt.as_posix()}")
    second_half = [l for l in read_metrics(full) if json.loads(l)["step"] > 100]
    assert len(second_half) == 100
    assert read_metrics(resumed) == second_half


def test_resume_ignores_current_seed(run_noop):
    """The checkpoint's RNG state wins over whatever seed the resumed run was given."""
    full = run_noop("seed=3")
    ckpt = checkpoint_path(full, 100).as_posix()
    resumed = run_noop("seed=99", f"run.resume_from={ckpt}")
    assert read_metrics(resumed) == [l for l in read_metrics(full) if json.loads(l)["step"] > 100]


def test_meta_records_seed_override(run_noop):
    full = run_noop("seed=3")
    meta_full = json.loads((full / "meta.json").read_text(encoding="utf-8"))
    assert meta_full["seed_overridden_by_checkpoint"] is False

    ckpt = checkpoint_path(full, 100).as_posix()
    resumed = run_noop("seed=99", f"run.resume_from={ckpt}")
    meta = json.loads((resumed / "meta.json").read_text(encoding="utf-8"))
    assert meta["seed_overridden_by_checkpoint"] is True
    assert meta["seed"] == 99  # the configured seed is still recorded, just not used
