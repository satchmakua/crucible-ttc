"""Beam/DVTS search: it needs a PRM, solves the stepwise task, and beats best-of-N."""

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


def test_beam_requires_a_process_verifier() -> None:
    cfg = _cfg("beam")
    cfg.prm = None  # no PRM → beam can't prune
    with pytest.raises(ValueError, match="needs a process verifier"):
        run(cfg)


def test_beam_solves_the_stepwise_task() -> None:
    summary = run(_cfg("beam", beam_width=6, beam_expansions=6, max_steps=8))
    assert summary.accuracy >= 0.8


def test_beam_beats_best_of_n() -> None:
    beam = run(_cfg("beam", beam_width=4, beam_expansions=4, max_steps=8))
    bon = run(_cfg("best_of_n", n=8, selection="prm"))
    assert beam.correct >= bon.correct


def test_beam_counts_policy_and_verifier_compute() -> None:
    c = run(_cfg("beam", beam_width=4, beam_expansions=4)).total_compute
    assert c.policy_forward_calls > 0
    assert c.verifier_forward_calls > 0
    assert c.verifier_gen_tokens > 0  # PRM forward-pass tokens land on the compute axis
