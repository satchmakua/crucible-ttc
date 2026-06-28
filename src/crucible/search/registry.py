"""Method name → `SearchStrategy`. The CLI's `--method` resolves through here."""

from __future__ import annotations

from crucible.domain.ports import SearchStrategy
from crucible.search.pass1 import Pass1Strategy

# Implemented now. Later milestones register best_of_n (M2), beam (M4), mcts (M6).
_STRATEGIES: dict[str, type[SearchStrategy]] = {
    "pass1": Pass1Strategy,
}

# Known-but-not-yet-built methods, so the CLI can give a milestone-aware message.
_PLANNED = {
    "best_of_n": "M2",
    "beam": "M4",
    "mcts": "M6",
}


def available_methods() -> list[str]:
    return sorted(_STRATEGIES)


def get_strategy(method: str) -> SearchStrategy:
    if method in _STRATEGIES:
        return _STRATEGIES[method]()
    if method in _PLANNED:
        raise NotImplementedError(
            f"method '{method}' is planned for milestone {_PLANNED[method]} "
            f"(see ROADMAP.md). Available now: {', '.join(available_methods())}."
        )
    raise ValueError(
        f"unknown method '{method}'. Available now: {', '.join(available_methods())}."
    )
