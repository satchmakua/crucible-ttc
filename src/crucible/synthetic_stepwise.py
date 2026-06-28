r"""A synthetic *stepwise* task to demonstrate PRM-guided beam search cold.

Real beam/DVTS beats best-of-N when a process verifier can prune bad partial reasoning
*early*, so compute concentrates on promising branches (Snell et al.). This module
fabricates exactly that regime without a model:

- `StepwisePolicy` models a `step_depth`-step process where each step is "good" with
  probability `step_accuracy`. The final answer is the problem's gold answer **iff every
  step was good**, else a distractor — so a trace is correct only if the whole chain is
  good (success probability ``p**depth``, which best-of-N must pay for exponentially).
- `StepRewardPRM` scores a partial trace by the quality of its steps (it reads the
  GOOD/BAD markers), so beam can keep all-good partials and assemble a correct chain
  from good steps across branches — linear, not exponential, in depth.

Both are seeded simulators (demo/test fixtures), paired here because they share the
step-marker convention.
"""

from __future__ import annotations

import random

from crucible.domain.types import Compute, Problem, Step, Trace
from crucible.segment import approx_tokens

GOOD_MARKER = "[GOOD]"
BAD_MARKER = "[BAD]"


def _distractor(gold: str | None) -> str:
    if gold is None:
        return "0"
    return "0" if gold.strip() != "0" else "1"


class StepwisePolicy:
    """A simulated D-step reasoner: emits the gold answer only if every step is good."""

    name = "stepwise"

    def __init__(self, *, step_accuracy: float = 0.6, depth: int = 4, seed: int = 0) -> None:
        self.step_accuracy = step_accuracy
        self.depth = depth
        self._rng = random.Random(seed)

    def _step_text(
        self, idx: int, good: bool, *, problem: Problem, prefix_all_good: bool, final: bool
    ) -> str:
        marker = GOOD_MARKER if good else BAD_MARKER
        quality = "sound" if good else "flawed"
        text = f"Step {idx + 1}: a {quality} reasoning step. {marker}"
        if final:
            all_good = prefix_all_good and good
            ans = problem.answer if (all_good and problem.answer is not None) else _distractor(
                problem.answer
            )
            text += f"\n\nFinal answer: \\boxed{{{ans}}}"
        return text

    def sample_step(
        self,
        problem: Problem,
        prefix: list[Step],
        *,
        n: int,
        temperature: float,
        max_tokens: int,
    ) -> list[Step]:
        idx = len(prefix)
        final = idx + 1 >= self.depth
        prefix_all_good = all(GOOD_MARKER in s.text for s in prefix)
        steps: list[Step] = []
        for _ in range(n):
            good = self._rng.random() < self.step_accuracy
            text = self._step_text(
                idx, good, problem=problem, prefix_all_good=prefix_all_good, final=final
            )
            steps.append(Step(text=text, token_count=approx_tokens(text)))
        return steps

    def sample_full(
        self, problem: Problem, *, n: int, temperature: float, max_tokens: int
    ) -> list[Trace]:
        traces: list[Trace] = []
        for _ in range(n):
            steps: list[Step] = []
            prefix_good = True
            for idx in range(self.depth):
                good = self._rng.random() < self.step_accuracy
                final = idx + 1 >= self.depth
                text = self._step_text(
                    idx, good, problem=problem, prefix_all_good=prefix_good, final=final
                )
                steps.append(Step(text=text, token_count=approx_tokens(text)))
                prefix_good = prefix_good and good
            compute = Compute(
                policy_gen_tokens=sum(s.token_count for s in steps), policy_forward_calls=1
            )
            traces.append(Trace(steps=steps, final_answer=None, compute=compute))
        return traces


class StepRewardPRM:
    """A simulated step-level PRM: scores steps by their GOOD/BAD marker, with skill."""

    name = "step-prm"

    def __init__(self, *, accuracy: float = 0.95, seed: int = 0) -> None:
        self.accuracy = accuracy
        self.seed = seed

    def score_steps(self, problem: Problem, prefix: list[Step]) -> list[float]:
        scores: list[float] = []
        spread = (1.0 - self.accuracy) * 0.5
        for i, step in enumerate(prefix):
            good = GOOD_MARKER in step.text
            rng = random.Random(f"{self.seed}:{problem.id}:{i}:{step.text}")
            base = 0.5 + (0.4 if good else -0.4) * self.accuracy
            scores.append(base + rng.uniform(-spread, spread))
        return scores or [0.0]
