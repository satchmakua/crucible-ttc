"""ScriptedPolicy — a deterministic, dependency-free `PolicyModel`.

It replays canned completion texts keyed by problem id. This is what makes the whole
engine testable without a model: the search core, verifier, runner, and reporting all
exercise real code paths, while generation is a lookup. Compute is reported honestly
(approximate token counts, one forward call per sample) so accounting is exercised too.
"""

from __future__ import annotations

from crucible.domain.types import Compute, Problem, Step, Trace
from crucible.segment import approx_tokens, segment


class ScriptedPolicy:
    """Replays pre-baked traces. `scripts` maps problem id → list of completions."""

    name = "mock"

    def __init__(self, scripts: dict[str, list[str]], *, max_step_tokens: int = 512) -> None:
        self._scripts = scripts
        self._max_step_tokens = max_step_tokens

    def _trace_for(self, text: str) -> Trace:
        steps = segment(text, max_step_tokens=self._max_step_tokens)
        compute = Compute(policy_gen_tokens=approx_tokens(text), policy_forward_calls=1)
        return Trace(steps=steps, final_answer=None, compute=compute)

    def sample_full(
        self, problem: Problem, *, n: int, temperature: float, max_tokens: int
    ) -> list[Trace]:
        outputs = self._scripts.get(problem.id, [])
        if not outputs:
            # No script for this problem: emit empty traces so the run still completes
            # (the verifier will score them as misses).
            return [Trace(steps=[], final_answer=None, compute=Compute(policy_forward_calls=1))
                    for _ in range(n)]
        return [self._trace_for(outputs[i % len(outputs)]) for i in range(n)]

    def sample_step(
        self,
        problem: Problem,
        prefix: list[Step],
        *,
        n: int,
        temperature: float,
        max_tokens: int,
    ) -> list[Step]:
        # Step-wise sampling is exercised by beam/MCTS (M4+). For the scripted backend
        # we return the next real step of the first script, padded as needed.
        outputs = self._scripts.get(problem.id, [])
        steps = segment(outputs[0], max_step_tokens=self._max_step_tokens) if outputs else []
        nxt = steps[len(prefix)] if len(prefix) < len(steps) else Step(text="", token_count=0)
        return [nxt for _ in range(n)]
