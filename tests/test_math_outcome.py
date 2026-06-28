"""Math outcome verifier: symbolic equivalence, not string matching."""

from __future__ import annotations

import pytest

from crucible.domain.types import Compute, Problem, Step, Trace
from crucible.verify import MathOutcomeVerifier, math_equal


@pytest.mark.parametrize(
    ("pred", "gold", "equal"),
    [
        ("72", "72", True),
        ("1/4", "0.25", True),  # the symbolic-equivalence case
        ("0.5", "1/2", True),
        ("1", "1", True),
        ("48", "12", False),
        ("144", "124", False),
    ],
)
def test_math_equal(pred: str, gold: str, equal: bool) -> None:
    assert math_equal(pred, gold) is equal


def _trace(text: str) -> Trace:
    return Trace(steps=[Step(text=text, token_count=1)], final_answer=None, compute=Compute())


def test_verifier_correct_on_boxed_answer() -> None:
    v = MathOutcomeVerifier()
    problem = Problem(id="x", prompt="...", answer="72")
    assert v.verify(problem, _trace("the answer is \\boxed{72}")).correct


def test_verifier_uses_equivalence_not_string_match() -> None:
    v = MathOutcomeVerifier()
    problem = Problem(id="x", prompt="...", answer="0.25")
    assert v.verify(problem, _trace("\\boxed{1/4}")).correct


def test_verifier_marks_wrong_answer() -> None:
    v = MathOutcomeVerifier()
    problem = Problem(id="x", prompt="...", answer="12")
    assert not v.verify(problem, _trace("\\boxed{48}")).correct


def test_verifier_no_gold_is_miss() -> None:
    v = MathOutcomeVerifier()
    problem = Problem(id="x", prompt="...", answer=None)
    assert not v.verify(problem, _trace("\\boxed{1}")).correct
