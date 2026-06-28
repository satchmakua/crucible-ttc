"""Crucible — a verifier-guided reasoning engine.

Generate many reasoning traces from a small open model, score them with a verifier
(programmatic checker or a process-reward model), and *search* — best-of-N → beam →
MCTS over reasoning steps — to turn test-time compute into measured accuracy.

The public surface is intentionally small; see `crucible.domain` for the ports the
whole engine is built around, and `crucible.cli` for the command-line front door.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
