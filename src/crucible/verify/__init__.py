"""Verifiers: outcome (ground-truth-ish) and process (PRM) checks.

M0 ships the math outcome verifier (answer extraction + symbolic equivalence). The
code-execution outcome verifier (M5) and PRM process verifier (M3) slot in behind the
same ports.
"""

from __future__ import annotations

from crucible.verify.answer_extract import extract_final_answer
from crucible.verify.math_outcome import MathOutcomeVerifier, math_equal

__all__ = ["MathOutcomeVerifier", "extract_final_answer", "math_equal"]
