"""Sweeps: run a grid of experiments → the accuracy-vs-compute curve.

A sweep YAML is the base run config plus a ``grid:`` list. Each grid cell overrides the
base; any list-valued field in a cell (e.g. ``n: [4, 8, 16]``) is expanded into the
cartesian product of runs. Every run writes its own record under the sweep directory,
and the cells are aggregated into ``sweep.json`` + ``curve.png`` — the headline
artifact (DESIGN.md §1, §6.4).
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from crucible.config import RunConfig
from crucible.report import render_curve, write_run_record
from crucible.runner import RunSummary, run


def expand_grid(base: dict[str, Any], grid: list[dict[str, Any]]) -> list[RunConfig]:
    """Expand a base config + grid cells into concrete `RunConfig`s."""
    configs: list[RunConfig] = []
    for cell in grid:
        merged: dict[str, Any] = {**base, **cell}
        list_keys = [k for k, v in merged.items() if isinstance(v, list)]
        if not list_keys:
            configs.append(RunConfig.from_dict(merged))
            continue
        for combo in itertools.product(*(merged[k] for k in list_keys)):
            expanded = dict(merged)
            for key, value in zip(list_keys, combo, strict=True):
                expanded[key] = value
            configs.append(RunConfig.from_dict(expanded))
    return configs


def cell_metrics(summary: RunSummary) -> dict[str, Any]:
    """One row of the sweep table — accuracy and mean per-problem compute."""
    low, high = summary.accuracy_ci
    denom = summary.total or 1
    # The knob that varies along a method's line: beam width for beam, token budget for
    # mcts, else N samples.
    cfg = summary.config
    if cfg.method == "beam":
        knob = cfg.beam_width
    elif cfg.method == "mcts":
        knob = cfg.budget_tokens or 0
    else:
        knob = cfg.n
    return {
        "method": summary.config.method,
        "selection": summary.config.selection,
        "n": knob,
        "total": summary.total,
        "correct": summary.correct,
        "accuracy": summary.accuracy,
        "accuracy_ci_low": low,
        "accuracy_ci_high": high,
        "mean_tokens": summary.total_compute.total_tokens / denom,
    }


@dataclass
class SweepResult:
    sweep_dir: Path
    cells: list[dict[str, Any]]
    curve_path: Path


def run_sweep(config_path: str | Path) -> SweepResult:
    """Run every cell of a sweep config and write records + sweep.json + curve.png."""
    with open(config_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("sweep config must be a YAML mapping")

    grid = data.get("grid") or []
    if not grid:
        raise ValueError("sweep config needs a non-empty 'grid:' list")
    base = {str(k): v for k, v in data.items() if k != "grid"}
    output_dir = str(base.get("output_dir", "runs"))
    configs = expand_grid(base, list(grid))

    sweep_dir = Path(output_dir) / datetime.now().strftime("sweep-%Y-%m-%dT%H-%M-%S")
    sweep_dir.mkdir(parents=True, exist_ok=True)

    cells: list[dict[str, Any]] = []
    for i, cfg in enumerate(configs):
        summary = run(cfg)
        name = f"{i:03d}-{cfg.method}-{cfg.selection}-n{cfg.n}"
        write_run_record(summary, base_dir=sweep_dir, name=name)
        cells.append(cell_metrics(summary))

    (sweep_dir / "sweep.json").write_text(json.dumps(cells, indent=2), encoding="utf-8")
    curve_path = render_curve(cells, sweep_dir / "curve.png")
    return SweepResult(sweep_dir=sweep_dir, cells=cells, curve_path=curve_path)
