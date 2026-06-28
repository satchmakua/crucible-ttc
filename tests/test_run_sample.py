"""End-to-end M0 spine: CLI/runner → mock policy → math verifier → report.

This is the test that proves the whole vertical slice works cold, with no GPU and no
network. The bundled `sample` set is built so `pass1` on the mock backend yields a
deterministic 4/6 pass@1.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from crucible.cli import app
from crucible.config import PolicyConfig, RunConfig
from crucible.report import read_summary, write_run_record
from crucible.runner import run


def _sample_config() -> RunConfig:
    return RunConfig(
        method="pass1",
        dataset="sample",
        policy=PolicyConfig(backend="mock", model="scripted"),
    )


def test_run_sample_pass_at_1() -> None:
    summary = run(_sample_config())
    assert summary.total == 6
    assert summary.correct == 4
    assert summary.accuracy == pytest.approx(4 / 6)


def test_run_sample_per_problem_verdicts() -> None:
    summary = run(_sample_config())
    verdicts = {r.problem_id: r.correct for r in summary.results}
    # Correct: 1, 4, 5, 6 (incl. the 1/4 ≡ 0.25 equivalence). Wrong: 2, 3.
    assert verdicts == {"1": True, "2": False, "3": False, "4": True, "5": True, "6": True}


def test_run_accounts_compute() -> None:
    summary = run(_sample_config())
    c = summary.total_compute
    assert c.policy_forward_calls == 6  # one sample per problem
    assert c.verifier_forward_calls == 6  # one outcome check per problem
    assert c.total_tokens > 0


def test_write_and_read_record(tmp_path: Path) -> None:
    summary = run(_sample_config())
    run_dir = write_run_record(summary, base_dir=tmp_path)
    assert (run_dir / "config.json").exists()
    assert (run_dir / "results.csv").exists()
    data = read_summary(run_dir)
    assert data["correct"] == 4
    assert data["total"] == 6
    assert data["compute"]["verifier_forward_calls"] == 6


def test_cli_run_smoke() -> None:
    result = CliRunner().invoke(
        app, ["run", "--method", "pass1", "--dataset", "sample", "--policy", "mock", "--no-save"]
    )
    assert result.exit_code == 0, result.output
    assert "accuracy" in result.output


def test_cli_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert "crucible" in result.output


def test_cli_unknown_method_errors() -> None:
    result = CliRunner().invoke(app, ["run", "--method", "nope", "--no-save"])
    assert result.exit_code == 1
