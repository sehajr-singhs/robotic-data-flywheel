import json
import tempfile
from pathlib import Path

from datafly import FlywheelConfig, run_flywheel, save_results


def test_quick_loop_runs_end_to_end(tmp_path: Path):
    cfg = FlywheelConfig(
        out_dir=str(tmp_path),
        strategies=("none", "success_only", "relabel", "relabel_curated"),
    ).quick()
    summary = run_flywheel(cfg)

    assert set(summary["strategies"]) == set(cfg.strategies)
    for name, data in summary["strategies"].items():
        assert len(data["success_rate_mean"]) == cfg.iterations + 1
        assert len(data["success_rate_seeds"]) == cfg.seeds
        assert all(0.0 <= v <= 1.0 for v in data["success_rate_mean"])
    assert summary["report"]["rollouts"]
    assert "plot_trajs" in summary["report"]

    path = save_results(summary, tmp_path)
    loaded = json.loads(path.read_text())
    assert loaded["config"]["seed"] == cfg.seed


def test_quick_loop_is_deterministic():
    cfg = FlywheelConfig(strategies=("success_only",)).quick()
    a = run_flywheel(cfg)
    b = run_flywheel(cfg)
    assert a["strategies"]["success_only"]["success_rate_mean"] == \
           b["strategies"]["success_only"]["success_rate_mean"]
