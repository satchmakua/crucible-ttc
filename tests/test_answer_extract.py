"""Answer extraction: the small heuristic that feeds the math verifier."""

from __future__ import annotations

import pytest

from crucible.verify import extract_final_answer


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Therefore the answer is \\boxed{72}.", "72"),
        ("\\boxed{1/4}", "1/4"),
        ("So we get \\boxed{40} mph.", "40"),
        ("The final answer is 144.", "144"),
        ("first \\boxed{1} then \\boxed{2}", "2"),  # takes the last boxed
        ("after simplifying we get 12 cupcakes left", "12"),  # number fallback
        ("\\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),  # one level of nested braces
    ],
)
def test_extracts_expected(text: str, expected: str) -> None:
    assert extract_final_answer(text) == expected


@pytest.mark.parametrize("text", ["", "no number or boxed answer here"])
def test_returns_none_when_nothing_found(text: str) -> None:
    assert extract_final_answer(text) is None
