"""SyntheticPolicy: a seeded, reproducible simulator of a known accuracy."""

from __future__ import annotations

from crucible.domain.types import Problem
from crucible.inference.synthetic import SyntheticPolicy
from crucible.verify import MathOutcomeVerifier

_PROBLEM = Problem(id="x", prompt="?", answer="42")
_OUTCOME = MathOutcomeVerifier()


def _rate(accuracy: float, *, n: int, seed: int = 0) -> float:
    traces = SyntheticPolicy(accuracy=accuracy, seed=seed).sample_full(
        _PROBLEM, n=n, temperature=0.0, max_tokens=8
    )
    return sum(_OUTCOME.verify(_PROBLEM, t).correct for t in traces) / len(traces)


def test_same_seed_is_deterministic() -> None:
    a = SyntheticPolicy(accuracy=0.5, seed=7).sample_full(_PROBLEM, n=10, temperature=0, max_tokens=8)
    b = SyntheticPolicy(accuracy=0.5, seed=7).sample_full(_PROBLEM, n=10, temperature=0, max_tokens=8)
    assert [t.text for t in a] == [t.text for t in b]


def test_empirical_accuracy_tracks_parameter() -> None:
    assert 0.6 < _rate(0.7, n=500) < 0.8


def test_extremes() -> None:
    assert _rate(0.0, n=50) == 0.0
    assert _rate(1.0, n=50) == 1.0
