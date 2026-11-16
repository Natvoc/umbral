import json

from conftest import read_metrics


def test_same_seed_gives_identical_metrics(run_noop):
    a = run_noop("seed=7", "run.name=a")
    b = run_noop("seed=7", "run.name=b")
    assert (a / "metrics.jsonl").read_bytes() == (b / "metrics.jsonl").read_bytes()
    assert len(read_metrics(a)) == 200


def test_different_seed_gives_different_metrics(run_noop):
    a = run_noop("seed=1")
    b = run_noop("seed=2")
    assert read_metrics(a) != read_metrics(b)


def test_run_directory_is_complete(run_noop):
    run_dir = run_noop("seed=0")
    for name in ("config.yaml", "meta.json", "metrics.jsonl", "log.txt"):
        assert (run_dir / name).is_file(), name
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "completed"
    assert meta["seed"] == 0
    for key in ("git_commit", "git_dirty", "python", "numpy", "torch", "threads"):
        assert key in meta
    assert meta["threads"]["num_threads"] == 4
