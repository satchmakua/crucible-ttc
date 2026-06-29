"""Sweeps: grid expansion + an end-to-end run that produces a lift curve."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from crucible.sweep import expand_grid, run_sweep


def test_expand_grid_cartesian_product() -> None:
    base: dict[str, Any] = {"dataset": "sample", "policy": {"backend": "synthetic"}}
    grid: list[dict[str, Any]] = [
        {"method": "pass1"},
        {"method": "best_of_n", "selection": "oracle", "n": [2, 4, 8]},
    ]
    cfgs = expand_grid(base, grid)
    assert len(cfgs) == 1 + 3
    bon = [c for c in cfgs if c.method == "best_of_n"]
    assert sorted(c.n for c in bon) == [2, 4, 8]
    assert all(c.selection == "oracle" for c in bon)
    assert all(c.policy.backend == "synthetic" for c in cfgs)


def test_run_sweep_writes_curve_and_shows_lift(tmp_path: Path) -> None:
    cfg = {
        "dataset": "sample",
        "seed": 0,
        "synthetic_accuracy": 0.4,
        "policy": {"backend": "synthetic", "model": "sim"},
        "grid": [
            {"method": "pass1"},
            {"method": "best_of_n", "selection": "oracle", "n": [4, 16]},
        ],
        "output_dir": str(tmp_path),
    }
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    result = run_sweep(path)

    assert result.curve_path.exists()
    assert (result.sweep_dir / "sweep.json").exists()

    pass1 = next(c for c in result.cells if c["method"] == "pass1")
    oracle16 = next(c for c in result.cells if c["method"] == "best_of_n" and c["n"] == 16)
    # Oracle best-of-16 should recover far more than single-sample pass@1...
    assert oracle16["accuracy"] >= pass1["accuracy"]
    # ...and it spends roughly 16x the per-problem tokens to do so.
    assert oracle16["mean_tokens"] > pass1["mean_tokens"]


def test_multi_seed_sweep_pools_results(tmp_path: Path) -> None:
    cfg = {
        "dataset": "sample",
        "seeds": [0, 1, 2],
        "synthetic_accuracy": 0.5,
        "policy": {"backend": "synthetic", "model": "sim"},
        "grid": [{"method": "pass1"}],
        "output_dir": str(tmp_path),
    }
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    result = run_sweep(path)
    cell = result.cells[0]
    assert cell["seeds"] == 3
    assert cell["total"] == 6 * 3  # 6 sample problems × 3 seeds, pooled
