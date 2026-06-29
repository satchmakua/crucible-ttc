"""Compute-optimal frontier + per-difficulty aggregation."""

from __future__ import annotations

from typing import Any

from crucible.analyze import accuracy_by_difficulty, compute_optimal_frontier
from crucible.domain.types import Compute, Result


def _cell(method: str, tokens: float, acc: float) -> dict[str, Any]:
    return {"method": method, "selection": "x", "n": 1, "mean_tokens": tokens, "accuracy": acc}


def test_frontier_keeps_non_dominated_points() -> None:
    cells = [
        _cell("pass1", 30, 0.1),
        _cell("best_of_n", 300, 0.5),
        _cell("best_of_n", 600, 0.4),  # dominated: pricier and worse than the 300 cell
        _cell("beam", 1000, 0.9),
        _cell("mcts", 5000, 0.9),  # dominated: same accuracy, far pricier
    ]
    frontier = compute_optimal_frontier(cells)
    labels = [(c["method"], c["mean_tokens"]) for c in frontier]
    assert labels == [("pass1", 30), ("best_of_n", 300), ("beam", 1000)]


def test_frontier_is_monotone_in_accuracy() -> None:
    cells = [_cell("a", 10, 0.2), _cell("b", 100, 0.6), _cell("c", 1000, 0.95)]
    accs = [c["accuracy"] for c in compute_optimal_frontier(cells)]
    assert accs == sorted(accs)


def _result(difficulty: str, correct: bool) -> Result:
    return Result(
        problem_id="p",
        method="m",
        dataset="d",
        correct=correct,
        predicted=None,
        gold=None,
        compute=Compute(),
        difficulty=difficulty,
    )


def test_accuracy_by_difficulty() -> None:
    results = [
        _result("easy", True),
        _result("easy", True),
        _result("hard", True),
        _result("hard", False),
    ]
    by = accuracy_by_difficulty(results)
    assert by["easy"]["accuracy"] == 1.0
    assert by["hard"]["accuracy"] == 0.5
    assert by["hard"]["total"] == 2.0
