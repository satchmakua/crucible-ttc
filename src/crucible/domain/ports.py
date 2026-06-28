"""The ports: the protocols the search/verification core depends on.

A new inference backend, verifier, or search strategy is an *adapter* that satisfies
one of these protocols — never a change to the core. Strategies consume the same
`PolicyModel` + verifier protocols, which is exactly what lets best-of-N, beam, and
MCTS share one generator and one verifier (DESIGN.md §4.3).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from crucible.domain.types import Problem, Step, Trace, Verdict

if TYPE_CHECKING:  # avoid a runtime import cycle with config
    from crucible.config import RunConfig


@runtime_checkable
class PolicyModel(Protocol):
    """A frozen generator behind the inference adapter (mock/Ollama/hosted/vLLM)."""

    name: str

    def sample_full(
        self, problem: Problem, *, n: int, temperature: float, max_tokens: int
    ) -> list[Trace]:
        """Sample `n` complete reasoning traces for `problem`."""
        ...

    def sample_step(
        self,
        problem: Problem,
        prefix: list[Step],
        *,
        n: int,
        temperature: float,
        max_tokens: int,
    ) -> list[Step]:
        """Sample `n` candidate next steps given a reasoning `prefix` (beam/MCTS)."""
        ...


@runtime_checkable
class OutcomeVerifier(Protocol):
    """Ground-truth-ish check: math equivalence or code execution. Pass/fail."""

    name: str

    def verify(self, problem: Problem, trace: Trace) -> Verdict: ...


@runtime_checkable
class ProcessVerifier(Protocol):
    """A PRM: a scalar score per step (no gold answer needed). Guides search only."""

    name: str

    def score_steps(self, problem: Problem, prefix: list[Step]) -> list[float]: ...


@runtime_checkable
class SearchStrategy(Protocol):
    """best-of-N | beam | mcts — all consume the same policy + verifier."""

    name: str

    def search(
        self,
        problem: Problem,
        policy: PolicyModel,
        outcome: OutcomeVerifier,
        process: ProcessVerifier | None,
        config: RunConfig,
    ) -> Trace:
        """Spend the configured budget searching for the best trace it can find.

        The returned trace's `.compute` must account for the *entire* search —
        every policy sample and every verifier call made along the way.
        """
        ...
