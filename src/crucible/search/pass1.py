"""pass1 — sample a single trace. The verifier-free baseline (pass@1).

The simplest rung of the ladder: no search at all. Every richer strategy is measured
against this line, so getting the plumbing exactly right here (one sample, honest
compute) matters.
"""

from __future__ import annotations

from crucible.config import RunConfig
from crucible.domain.ports import OutcomeVerifier, PolicyModel, ProcessVerifier
from crucible.domain.types import Compute, Problem, Trace


class Pass1Strategy:
    """Draw one sample and return it unchanged."""

    name = "pass1"

    def search(
        self,
        problem: Problem,
        policy: PolicyModel,
        outcome: OutcomeVerifier,
        process: ProcessVerifier | None,
        config: RunConfig,
    ) -> Trace:
        traces = policy.sample_full(
            problem,
            n=1,
            temperature=config.policy.temperature,
            max_tokens=config.policy.max_tokens,
        )
        return traces[0] if traces else Trace(steps=[], final_answer=None, compute=Compute())
