"""Cassettes: record real model calls once, replay them offline forever (ROADMAP H3).

The standing pattern (CLAUDE.md / DESIGN §9): **run live once, then commit a recorded
fixture so CI reproduces the numbers without a GPU.** `RecordingPolicy` wraps a real
backend (e.g. Ollama) and captures every problem → traces into a self-contained JSON
cassette (the problems *and* their generated traces). `CassettePolicy` replays that
cassette with no network and no model, so a real run's pass@1 regenerates deterministically.

Only the policy (generation) side is cassetted here; the PRM (process-verifier) side gets
the same treatment once a GPU run records it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from crucible.domain.ports import PolicyModel
from crucible.domain.types import Compute, Problem, Step, Trace


def _trace_to_dict(trace: Trace) -> dict[str, Any]:
    c = trace.compute
    return {
        "steps": [{"text": s.text, "token_count": s.token_count} for s in trace.steps],
        "compute": {
            "policy_gen_tokens": c.policy_gen_tokens,
            "policy_forward_calls": c.policy_forward_calls,
            "wall_seconds": c.wall_seconds,
        },
    }


def _trace_from_dict(data: dict[str, Any]) -> Trace:
    steps = [Step(text=s["text"], token_count=int(s["token_count"])) for s in data["steps"]]
    c = data.get("compute", {})
    compute = Compute(
        policy_gen_tokens=int(c.get("policy_gen_tokens", 0)),
        policy_forward_calls=int(c.get("policy_forward_calls", 0)),
        wall_seconds=float(c.get("wall_seconds", 0.0)),
    )
    return Trace(steps=steps, final_answer=None, compute=compute)


def _problem_to_dict(p: Problem) -> dict[str, Any]:
    return {"id": p.id, "prompt": p.prompt, "answer": p.answer, "difficulty": p.difficulty}


def _problem_from_dict(data: dict[str, Any]) -> Problem:
    return Problem(
        id=str(data["id"]),
        prompt=str(data["prompt"]),
        answer=data.get("answer"),
        difficulty=data.get("difficulty"),
    )


class RecordingPolicy:
    """Wraps a real `PolicyModel`, forwarding calls and recording problem → traces."""

    name = "recording"

    def __init__(self, inner: PolicyModel, path: str | Path) -> None:
        self._inner = inner
        self._path = Path(path)
        self._records: dict[str, tuple[Problem, list[Trace]]] = {}

    def sample_full(
        self, problem: Problem, *, n: int, temperature: float, max_tokens: int
    ) -> list[Trace]:
        traces = self._inner.sample_full(
            problem, n=n, temperature=temperature, max_tokens=max_tokens
        )
        self._records[problem.id] = (problem, traces)
        return traces

    def sample_step(
        self, problem: Problem, prefix: list[Step], *, n: int, temperature: float, max_tokens: int
    ) -> list[Step]:
        return self._inner.sample_step(
            problem, prefix, n=n, temperature=temperature, max_tokens=max_tokens
        )

    def save(self) -> Path:
        """Write the captured problems + traces to a self-contained JSON cassette."""
        data = {
            "backend": getattr(self._inner, "name", "unknown"),
            "records": [
                {"problem": _problem_to_dict(p), "traces": [_trace_to_dict(t) for t in ts]}
                for p, ts in self._records.values()
            ],
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return self._path


class CassettePolicy:
    """Replays a recorded cassette — no network, no model, fully deterministic."""

    name = "cassette"

    def __init__(self, records: dict[str, list[Trace]]) -> None:
        self._records = records

    def sample_full(
        self, problem: Problem, *, n: int, temperature: float, max_tokens: int
    ) -> list[Trace]:
        traces = self._records.get(problem.id)
        if not traces:
            return [Trace(steps=[], final_answer=None, compute=Compute()) for _ in range(n)]
        return [traces[i % len(traces)] for i in range(n)]

    def sample_step(
        self, problem: Problem, prefix: list[Step], *, n: int, temperature: float, max_tokens: int
    ) -> list[Step]:
        return [Step(text="", token_count=0) for _ in range(n)]


def load_cassette(path: str | Path) -> tuple[list[Problem], dict[str, list[Trace]]]:
    """Load a cassette into (problems, id → traces) for offline replay."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    problems: list[Problem] = []
    records: dict[str, list[Trace]] = {}
    for rec in data["records"]:
        problem = _problem_from_dict(rec["problem"])
        problems.append(problem)
        records[problem.id] = [_trace_from_dict(t) for t in rec["traces"]]
    return problems, records
