"""The domain: pure value types and the ports (protocols) the engine depends on.

Nothing in here imports an inference backend, a dataset library, or a plotting
library. The search/verification core depends *only* on these definitions, which is
what keeps backends, verifiers, datasets, and reporting swappable (see
`docs/adr/0002-ports-and-adapters-with-compute-accounting.md` and DESIGN.md §4–§5).
"""

from __future__ import annotations

from crucible.domain.ports import (
    OutcomeVerifier,
    PolicyModel,
    ProcessVerifier,
    SearchStrategy,
)
from crucible.domain.types import Compute, Problem, Result, Step, Trace, Verdict

__all__ = [
    "Compute",
    "OutcomeVerifier",
    "PolicyModel",
    "Problem",
    "ProcessVerifier",
    "Result",
    "SearchStrategy",
    "Step",
    "Trace",
    "Verdict",
]
