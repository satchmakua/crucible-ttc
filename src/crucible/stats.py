"""Small statistics helpers. Wilson confidence intervals for reported accuracy.

A normal-approximation interval is wrong at the small-N, near-0/1 accuracies this
harness routinely produces; Wilson behaves. (Matches the stats in `llm-eval-harness`.)
"""

from __future__ import annotations

import math


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (default 95%)."""
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))
