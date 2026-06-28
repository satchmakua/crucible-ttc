"""MCTS over the step tree: needs a PRM, solves the stepwise task, counts compute."""

from __future__ import annotations

from typing import Any

import pytest

from crucible.config import PolicyConfig, RunConfig
from crucible.runner import run


def _cfg(method: str, **kw: Any) -> RunConfig:
    base: dict[str, Any] = {
        "method": method,
        "dataset": "sample",
        "seed": 0,
        "prm": "step",
        "step_accuracy": 0.6,
        "step_depth": 5,
        "step_prm_accuracy": 0.99,
        "policy": PolicyConfig(backend="stepwise", model="sim"),
    }
    base.update(kw)
    return RunConfig(**base)


def test_mcts_requires_a_process_verifier() -> None:
    cfg = _cfg("mcts", budget_tokens=2000)
    cfg.prm = None
    with pytest.raises(ValueError, match="needs a process verifier"):
        run(cfg)


def test_mcts_solves_the_stepwise_task_with_enough_budget() -> None:
    # MCTS saturates this (easy, shallow) task — it just needs more compute than beam,
    # which is the honest result: tree-search overhead only pays off on harder problems.
    summary = run(_cfg("mcts", budget_tokens=6000, beam_expansions=4))
    assert summary.accuracy >= 0.8


def test_mcts_respects_token_budget() -> None:
    small = run(_cfg("mcts", budget_tokens=1500, beam_expansions=4))
    large = run(_cfg("mcts", budget_tokens=6000, beam_expansions=4))
    # A bigger budget spends more tokens and never does worse on this task.
    assert large.total_compute.total_tokens > small.total_compute.total_tokens
    assert large.accuracy >= small.accuracy


def test_mcts_counts_policy_and_verifier_compute() -> None:
    c = run(_cfg("mcts", budget_tokens=2000)).total_compute
    assert c.policy_forward_calls > 0
    assert c.verifier_forward_calls > 0
    assert c.verifier_gen_tokens > 0  # PRM forward-pass tokens land on the compute axis
