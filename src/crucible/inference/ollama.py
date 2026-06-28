"""OllamaPolicy — the default real backend (native Windows).

Talks to a running Ollama server's HTTP API. Present from M0 so the port is real, but
the M0 verified path uses the mock backend; this is exercised from M1 (the first run
against live GSM8K). Token counts come straight from Ollama's `eval_count` so compute
accounting stays honest.
"""

from __future__ import annotations

import os
import time

import httpx

from crucible.domain.types import Compute, Problem, Step, Trace
from crucible.segment import approx_tokens, segment

DEFAULT_HOST = "http://localhost:11434"


class OllamaPolicy:
    """`PolicyModel` backed by an Ollama server (`/api/generate`)."""

    name = "ollama"

    def __init__(
        self,
        model: str,
        *,
        host: str | None = None,
        max_step_tokens: int = 512,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
        self._max_step_tokens = max_step_tokens
        self._timeout = timeout

    def _generate(self, prompt: str, *, temperature: float, max_tokens: int) -> tuple[str, int]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        resp = httpx.post(f"{self.host}/api/generate", json=payload, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()
        text = str(data.get("response", ""))
        tokens = int(data.get("eval_count") or approx_tokens(text))
        return text, tokens

    def _trace(self, text: str, tokens: int, elapsed: float) -> Trace:
        steps = segment(text, max_step_tokens=self._max_step_tokens)
        compute = Compute(
            policy_gen_tokens=tokens, policy_forward_calls=1, wall_seconds=elapsed
        )
        return Trace(steps=steps, final_answer=None, compute=compute)

    def sample_full(
        self, problem: Problem, *, n: int, temperature: float, max_tokens: int
    ) -> list[Trace]:
        traces: list[Trace] = []
        for _ in range(n):
            t0 = time.perf_counter()
            text, tokens = self._generate(
                problem.prompt, temperature=temperature, max_tokens=max_tokens
            )
            traces.append(self._trace(text, tokens, time.perf_counter() - t0))
        return traces

    def sample_step(
        self,
        problem: Problem,
        prefix: list[Step],
        *,
        n: int,
        temperature: float,
        max_tokens: int,
    ) -> list[Step]:
        prefix_text = "\n\n".join(s.text for s in prefix)
        prompt = f"{problem.prompt}\n\n{prefix_text}".strip()
        steps: list[Step] = []
        for _ in range(n):
            text, tokens = self._generate(
                prompt, temperature=temperature, max_tokens=min(max_tokens, self._max_step_tokens)
            )
            first = segment(text, max_step_tokens=self._max_step_tokens)
            steps.append(first[0] if first else Step(text=text, token_count=tokens))
        return steps
