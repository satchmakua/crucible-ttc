"""Step segmentation and the token approximation used for compute accounting."""

from __future__ import annotations

from crucible.segment import approx_tokens, segment


def test_splits_on_blank_lines() -> None:
    steps = segment("first step\n\nsecond step\n\nthird step")
    assert [s.text for s in steps] == ["first step", "second step", "third step"]


def test_ignores_empty_chunks_and_strips() -> None:
    steps = segment("\n\n  alpha  \n\n\n\n beta \n\n")
    assert [s.text for s in steps] == ["alpha", "beta"]


def test_enforces_token_cap() -> None:
    long_chunk = " ".join(str(i) for i in range(10))
    steps = segment(long_chunk, max_step_tokens=4)
    assert len(steps) == 3  # 4 + 4 + 2
    assert all(s.token_count <= 4 for s in steps)


def test_approx_tokens() -> None:
    assert approx_tokens("one two three") == 3
    assert approx_tokens("   ") == 0
    assert approx_tokens("solo") == 1
