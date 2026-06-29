"""Post-hoc analysis over run/sweep results.

Two views the report leans on:

- **The compute-optimal frontier** — the upper-left envelope of accuracy vs total
  tokens across *all* methods. At each compute budget it answers "what's the best
  accuracy any method gets for this many tokens, and which method?" This is the
  compute-optimal-scaling result (Snell et al.): the best method shifts with budget.
- **Accuracy by difficulty** — the same runs bucketed by problem difficulty, since the
  compute-optimal choice depends on how hard the problem is.
"""

from __future__ import annotations

from typing import Any

from crucible.domain.types import Result


def compute_optimal_frontier(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The non-dominated points: cheapest cell achieving each new best accuracy.

    Walk cells cheapest-first; keep a cell only if it strictly beats the best accuracy
    seen at lower compute. The result is the staircase of "best accuracy achievable at
    or below this many tokens", tagged with the method that achieves it.
    """
    ordered = sorted(cells, key=lambda c: (c["mean_tokens"], -c["accuracy"]))
    frontier: list[dict[str, Any]] = []
    best_acc = float("-inf")
    for cell in ordered:
        if cell["accuracy"] > best_acc:
            frontier.append(cell)
            best_acc = cell["accuracy"]
    return frontier


def accuracy_by_difficulty(results: list[Result]) -> dict[str, dict[str, float]]:
    """Bucket per-problem results by difficulty → {correct, total, accuracy}."""
    buckets: dict[str, list[Result]] = {}
    for r in results:
        buckets.setdefault(r.difficulty or "unknown", []).append(r)
    out: dict[str, dict[str, float]] = {}
    for difficulty, rows in sorted(buckets.items()):
        total = len(rows)
        correct = sum(1 for r in rows if r.correct)
        out[difficulty] = {
            "correct": float(correct),
            "total": float(total),
            "accuracy": correct / total if total else 0.0,
        }
    return out
