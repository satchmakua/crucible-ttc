r"""Pull a final answer out of a free-form reasoning trace.

This is the small module that feeds the math verifier (DESIGN.md §4.4/§6.2). It tries,
in order: the last ``\boxed{...}``, an "answer is/answer:" phrase, then the last
number in the text. Returns ``None`` if it finds nothing — the verifier treats that
as a miss rather than guessing.
"""

from __future__ import annotations

import re

# \boxed{...} with one level of nested braces tolerated (e.g. \boxed{\frac{1}{2}}).
_BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
_ANSWER_PHRASE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|:|=)?\s*\$?\\?boxed?\s*\{?\s*([^\n.$}]+)",
    re.IGNORECASE,
)
# Signed integers, decimals, and simple fractions like -3/4 or 1.5.
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+(?:\.\d+)?)?")


def _clean(s: str) -> str:
    return s.strip().strip("$").strip().rstrip(".").strip()


def extract_final_answer(text: str) -> str | None:
    """Best-effort extraction of the final answer string from `text`."""
    if not text:
        return None

    boxed = _BOXED.findall(text)
    if boxed:
        cleaned = _clean(boxed[-1])
        if cleaned:
            return cleaned

    phrase = _ANSWER_PHRASE.findall(text)
    if phrase:
        cleaned = _clean(phrase[-1])
        if cleaned:
            return cleaned

    numbers = _NUMBER.findall(text)
    if numbers:
        cleaned = _clean(numbers[-1]).replace(" ", "")
        if cleaned:
            return cleaned

    return None
