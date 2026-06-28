"""Step segmentation and token approximation.

A "step" is the unit search branches on. The default boundary is a blank line
(`\\n\\n`), with a hard per-step token cap so a runaway step can't blow up the search
tree (DESIGN.md §4.1). Segmentation is configurable and recorded per run because it
materially changes beam/MCTS behaviour.

`approx_tokens` is a whitespace proxy used for the mock backend and for the per-step
cap; real inference adapters report true tokenizer counts in `Compute`.
"""

from __future__ import annotations

import re

from crucible.domain.types import Step

_BLANK_LINE = re.compile(r"\n\s*\n")


def approx_tokens(text: str) -> int:
    """A cheap, deterministic token estimate (whitespace-delimited words)."""
    n = len(text.split())
    return max(n, 1) if text.strip() else 0


def _cap(words: list[str], max_step_tokens: int) -> list[str]:
    """Split an over-long chunk into <= max_step_tokens word groups."""
    if max_step_tokens <= 0:
        return [" ".join(words)]
    return [" ".join(words[i : i + max_step_tokens]) for i in range(0, len(words), max_step_tokens)]


def segment(text: str, *, max_step_tokens: int = 512) -> list[Step]:
    """Split `text` into reasoning steps on blank lines, enforcing a token cap."""
    chunks = [c.strip() for c in _BLANK_LINE.split(text) if c.strip()]
    steps: list[Step] = []
    for chunk in chunks:
        words = chunk.split()
        if len(words) <= max_step_tokens:
            steps.append(Step(text=chunk, token_count=len(words)))
            continue
        for piece in _cap(words, max_step_tokens):
            steps.append(Step(text=piece, token_count=len(piece.split())))
    return steps
