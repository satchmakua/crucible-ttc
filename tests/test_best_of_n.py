"""best_of_n selection (majority / oracle) and honest compute accounting."""

from __future__ import annotations

import pytest

from crucible.config import PolicyConfig, RunConfig
from crucible.domain.types import Problem
from crucible.inference.mock import ScriptedPolicy
from crucible.search.best_of_n import BestOfNStrategy
from crucible.verify import MathOutcomeVerifier, extract_final_answer

_PROBLEM = Problem(id="p", prompt="what is it?", answer="2")
# Candidate completions: answers 1 (wrong), 2 (right), 2 (right). pass1 would pick 1.
_SCRIPTS = {"p": ["reason \\boxed{1}", "reason \\boxed{2}", "reason \\boxed{2}"]}
_POLICY = ScriptedPolicy(_SCRIPTS)
_OUTCOME = MathOutcomeVerifier()


def _cfg(n: int, selection: str) -> RunConfig:
    return RunConfig(
        method="best_of_n", n=n, selection=selection, policy=PolicyConfig(backend="mock")
    )


def test_majority_picks_the_modal_answer() -> None:
    chosen = BestOfNStrategy().search(_PROBLEM, _POLICY, _OUTCOME, None, _cfg(3, "majority"))
    assert extract_final_answer(chosen.text) == "2"
    assert _OUTCOME.verify(_PROBLEM, chosen).correct


def test_oracle_finds_a_correct_trace_and_counts_checks() -> None:
    chosen = BestOfNStrategy().search(_PROBLEM, _POLICY, _OUTCOME, None, _cfg(3, "oracle"))
    assert _OUTCOME.verify(_PROBLEM, chosen).correct
    # Stopped at the 2nd candidate (first one that passed).
    assert chosen.compute.verifier_forward_calls == 2


def test_compute_accounts_all_samples() -> None:
    chosen = BestOfNStrategy().search(_PROBLEM, _POLICY, _OUTCOME, None, _cfg(3, "majority"))
    assert chosen.compute.policy_forward_calls == 3
    assert chosen.compute.policy_gen_tokens > 0


def test_best_of_n_beats_pass1_on_this_problem() -> None:
    # pass1 takes the first sample (answer 1, wrong); best_of_n recovers the right one.
    first = _POLICY.sample_full(_PROBLEM, n=1, temperature=0.0, max_tokens=8)[0]
    assert not _OUTCOME.verify(_PROBLEM, first).correct
    chosen = BestOfNStrategy().search(_PROBLEM, _POLICY, _OUTCOME, None, _cfg(3, "oracle"))
    assert _OUTCOME.verify(_PROBLEM, chosen).correct


def test_unknown_selection_raises() -> None:
    with pytest.raises(ValueError, match="unknown selection"):
        BestOfNStrategy().search(_PROBLEM, _POLICY, _OUTCOME, None, _cfg(3, "bogus"))
