"""Inference adapters: one `PolicyModel` port, several backends.

- `ScriptedPolicy` (mock) — deterministic replay of canned traces; no GPU, no
  network. The M0 spine and the test suite run on this.
- `OllamaPolicy` — the default real backend (native Windows), used from M1.

vLLM and a hosted OpenAI-compatible adapter slot in behind the same port later.
"""

from __future__ import annotations

from crucible.inference.mock import ScriptedPolicy
from crucible.inference.ollama import OllamaPolicy

__all__ = ["OllamaPolicy", "ScriptedPolicy"]
