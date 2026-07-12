"""Cassette record/replay (H3): a run's generations reproduce offline in CI."""

from __future__ import annotations

from pathlib import Path

from crucible.config import PolicyConfig, RunConfig
from crucible.inference import CassettePolicy, load_cassette
from crucible.runner import run
from crucible.verify import MathOutcomeVerifier

_OUTCOME = MathOutcomeVerifier()


def _replay_pass_at_1(problems: list, records: dict) -> int:
    policy = CassettePolicy(records)
    return sum(
        _OUTCOME.verify(p, policy.sample_full(p, n=1, temperature=0.0, max_tokens=1)[0]).correct
        for p in problems
    )


def test_record_then_replay_reproduces_results(tmp_path: Path) -> None:
    cassette = tmp_path / "sample.json"
    # Record a run (mock backend stands in for a real one), then replay it offline.
    cfg = RunConfig(
        method="pass1",
        dataset="sample",
        record=str(cassette),
        policy=PolicyConfig(backend="mock", model="scripted"),
    )
    recorded = run(cfg)
    assert cassette.exists()

    problems, records = load_cassette(cassette)
    assert len(problems) == recorded.total
    # The offline replay reproduces the recorded run's pass@1 exactly — no model needed.
    assert _replay_pass_at_1(problems, records) == recorded.correct == 4


# --- Real captured fixture (H3): a live GSM8K run, replayed with no GPU/network. -----

_FIXTURE = Path(__file__).parent / "fixtures" / "gsm8k-m1.json"


def test_real_gsm8k_fixture_reproduces_pass_at_1() -> None:
    if not _FIXTURE.exists():  # pragma: no cover - present once a real run is recorded
        import pytest

        pytest.skip("no recorded GSM8K fixture yet — run `crucible run ... --record`")
    problems, records = load_cassette(_FIXTURE)
    assert len(problems) == 3
    # Reproduces the real Ollama run's numbers offline, no GPU/network (see PROGRESS):
    # qwen2.5:7b-instruct, greedy, 3/3 on the first 3 GSM8K test problems.
    assert _replay_pass_at_1(problems, records) == 3
