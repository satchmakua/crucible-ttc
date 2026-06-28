"""Dataset name → problems (and, for the mock backend, scripted outputs)."""

from __future__ import annotations

from crucible.data.sample import SAMPLE_PROBLEMS, SAMPLE_SCRIPTS
from crucible.domain.types import Problem

# Real datasets land in M1 via HuggingFace `datasets`; named here so the CLI can give
# a milestone-aware message instead of a bare KeyError.
_HF_DATASETS = {
    "gsm8k": "M1",
    "math500": "M1",
    "humaneval": "M5",
    "mbpp": "M5",
}


def available_datasets() -> list[str]:
    return ["sample"]


def load_dataset(name: str, *, limit: int | None = None) -> list[Problem]:
    if name == "sample":
        problems = list(SAMPLE_PROBLEMS)
    elif name in _HF_DATASETS:
        raise NotImplementedError(
            f"dataset '{name}' loads via the HuggingFace `datasets` extra in milestone "
            f"{_HF_DATASETS[name]} (see ROADMAP.md). Available now: "
            f"{', '.join(available_datasets())}."
        )
    else:
        raise ValueError(
            f"unknown dataset '{name}'. Available now: {', '.join(available_datasets())}."
        )
    return problems[:limit] if limit is not None else problems


def scripts_for(name: str) -> dict[str, list[str]]:
    """Canned mock-backend outputs for a dataset (only the bundled `sample` set has them)."""
    return dict(SAMPLE_SCRIPTS) if name == "sample" else {}
